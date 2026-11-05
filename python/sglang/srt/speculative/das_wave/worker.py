"""Native V2 Wave policy: packed causal positions, independent R4 verifier.

One native call executes a complete block for every selected request. All
positions/requests share projection and attention launches. Guesses and hidden
states stay on device across refinement rounds; a single small decision packet
is read back after target verification to settle native allocation ownership.
"""

import logging
from dataclasses import dataclass

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.recurrent_runner import RecurrentRunner, RecurrentState
from sglang.srt.runtime_context import get_spec
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.loopspec.utils import split_batch
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils.device_timer import device_timer_ctx
from sglang.srt.utils.recurrent_trace import span

from .config import wave_widths
from .info import DASWaveInput

logger = logging.getLogger(__name__)


@dataclass
class WaveContext:
    req: object
    row: int
    length: int
    anchor: torch.Tensor
    emitted: int
    cycles: int = 0
    accepted: int = 0
    tested: int = 0
    proposed: int = 0


class DASWaveWorker(BaseSpecWorker):
    def __init__(self, server_args, gpu_id, nccl_port, target_worker):
        super().__init__()
        self._target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.model = self.model_runner.model
        self.device = self.model_runner.device
        self.server_args = server_args
        spec = get_spec()
        self.widths = wave_widths(spec.speculative_das_wave_max_width,
                                 spec.speculative_das_wave_rounds,
                                 spec.speculative_das_wave_initial_width)
        self.updates = spec.speculative_das_wave_updates
        self.width = self.widths[-1]
        self.total = self.model.config.total_ut_steps
        self.speculative_num_draft_tokens = spec.speculative_num_draft_tokens
        self.adaptive_controller = None
        self.contexts = {}
        self.temporary = None

    @property
    def recurrent_policy(self):
        return self

    @property
    def draft_worker(self):
        return None

    def alloc_memory_pool(self, **kwargs):
        self.req_to_token_pool, self.token_to_kv_pool_allocator = self.target_worker.get_memory_pool()
        self.max_rows = self.model_runner.max_running_requests * self.width
        self.private_pool = ReqToTokenPool(
            size=self.max_rows, max_context_len=self.req_to_token_pool.max_context_len,
            device=self.device, enable_memory_saver=False)
        self.private_pool.req_to_token.zero_()

    def init_attention_backends(self):
        if self.model_runner.recurrent_runner is None:
            raise RuntimeError("DAS-Wave requires the native recurrent executor")
        self.executor = RecurrentRunner(self.model_runner, max_rows=self.max_rows,
                                        request_pool=self.private_pool)

    def init_cuda_graphs(self):
        # Native prefill graphs remain owned by the target recurrent runner.
        # Wave graph capture is added only after the eager KV contract passes.
        pass

    def extra_owned_slots(self):
        return [] if self.temporary is None else self.temporary.tolist()

    def clear_cache_pool(self):
        if self.temporary is not None:
            raise RuntimeError("Cannot flush during an unfinished Wave transaction")
        self.contexts.clear()

    def note_request_retracted(self, *, rid):
        self.contexts.pop(rid, None)

    def note_request_finished(self, *, rid, natural_stop):
        c = self.contexts.pop(rid, None)
        if c is not None:
            logger.info("DAS_WAVE rid=%s cycles=%d proposed=%d tested=%d accepted=%d emitted=%d",
                        rid, c.cycles, c.proposed, c.tested, c.accepted, c.emitted)

    def split_recurrent_batch(self, batch):
        return split_batch(batch)

    def finish_recurrent_prefill(self, batch, result):
        for i, (req, row, length) in enumerate(zip(batch.reqs, batch.req_pool_indices.tolist(),
                                                  batch.seq_lens.tolist(), strict=True)):
            if req is batch.chunked_req or req.finished() or req.sampling_params.max_new_tokens <= 1:
                continue
            if req.rid in self.contexts:
                raise RuntimeError("Wave owner already exists before prefill")
            self.contexts[req.rid] = WaveContext(req, row, length,
                result.next_token_ids[i:i+1].clone(), len(req.output_ids) + 1)
        result.accept_lens = torch.ones_like(batch.seq_lens, dtype=torch.int32)
        result.new_seq_lens = batch.seq_lens
        result.speculative_num_draft_tokens = self.speculative_num_draft_tokens
        result.speculative_output_stride = 1
        result.next_draft_input = DASWaveInput(result.next_token_ids, batch.seq_lens)
        return result

    @span("das_wave.prepare")
    def _prepare(self, contexts, widths, slots, guesses):
        """Construct causal rows in batches; no token-wise scalar device reads."""
        owners = torch.repeat_interleave(torch.arange(len(contexts), device=self.device),
            torch.tensor(widths, device=self.device), output_size=sum(widths))
        columns = torch.cat([torch.arange(w, device=self.device) for w in widths])
        lengths = torch.tensor([c.length for c in contexts], device=self.device, dtype=torch.int32)
        prefix = lengths[owners]
        native_rows = torch.tensor([c.row for c in contexts], device=self.device)
        table = self.private_pool.req_to_token
        rows = torch.arange(1, len(owners) + 1, device=self.device)
        table[rows] = self.req_to_token_pool.req_to_token[native_rows[owners]]
        # Every position sees one common chain but a different causal endpoint.
        for owner, (c, w) in enumerate(zip(contexts, widths, strict=True)):
            offset = sum(widths[:owner]) + 1
            table[offset:offset+w, c.length:c.length+w] = slots[owner, :w]
        batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE, batch_size=len(owners),
            input_ids=guesses[owners, columns], req_pool_indices=rows,
            seq_lens=prefix + columns.to(torch.int32) + 1,
            seq_lens_sum=sum((c.length + 1) * w + w * (w-1)//2
                             for c, w in zip(contexts, widths, strict=True)),
            out_cache_loc=slots[owners, columns], positions=prefix.long() + columns,
            capture_hidden_mode=CaptureHiddenMode.NULL, spec_algorithm=SpeculativeAlgorithm.NONE)
        return batch, prefix, owners, columns

    @span("das_wave.draft_core")
    def _draft_core(self, batch, hidden, prefix):
        mapped, backend = self.executor.timeline.draft_batch(batch, prefix)
        with forward_context(ForwardContext(backend)):
            return self.model.advance_recurrent(hidden, batch.positions,
                                                 [(0, 0, len(hidden), mapped, backend)])

    @span("das_wave.readout")
    def _readout(self, batch, hidden):
        # Ouro has no attention-bearing coda. Return all packed position logits,
        # rather than EXTEND's last-token-only LogitsProcessor path.
        return self.model.readout_recurrent(hidden, batch.positions, [])

    @span("das_wave.draft")
    def _draft(self, contexts, caps, slots):
        anchors = torch.cat([c.anchor for c in contexts])
        guesses = anchors[:, None].expand(-1, self.width + 1).clone()
        hidden = torch.zeros((len(contexts), self.width, self.model.config.hidden_size),
                             dtype=self.model.get_input_embeddings().weight.dtype, device=self.device)
        previous = [0] * len(contexts)
        for width in self.widths:
            widths = [min(width, cap) for cap in caps]
            batch, prefix, owners, columns = self._prepare(contexts, widths, slots, guesses)
            fresh = columns >= torch.tensor(previous, device=self.device)[owners]
            initialized = self.model.get_input_embeddings()(batch.input_ids)
            carried = torch.where(fresh[:, None], initialized, hidden[owners, columns])
            for _ in range(self.updates):
                carried = self._draft_core(batch, carried, prefix)
            hidden[owners, columns] = carried
            guesses[owners, columns + 1] = self._readout(batch, carried).argmax(-1)
            previous = widths
        return guesses

    @span("das_wave.verify")
    def _verify(self, contexts, widths, slots, guesses):
        batch, _, owners, columns = self._prepare(contexts, widths, slots, guesses)
        hidden = self.model.get_input_embeddings()(batch.input_ids)
        for depth in range(self.total):
            state = self.executor.advance(batch, RecurrentState(hidden),
                                          torch.full_like(batch.req_pool_indices, depth))
            hidden = state.hidden
        logits = self._readout(batch, hidden)
        return logits, owners, columns

    @span("das_wave.decision")
    def _decision(self, contexts, widths, guesses, logits, owners, columns):
        predictions = torch.full((len(contexts), self.width), -1, dtype=torch.int64, device=self.device)
        predictions[owners, columns] = logits.argmax(-1)
        positions = torch.arange(self.width, device=self.device)[None, :]
        width_tensor = torch.tensor(widths, device=self.device)[:, None]
        mismatches = guesses[:, 1:] != predictions
        first_mismatch = torch.where(mismatches & (positions < width_tensor), positions, self.width).amin(-1)
        emitted = torch.minimum(first_mismatch + 1, width_tensor[:, 0])
        # EOS termination is part of the commit transaction, before KV transfer.
        for i, c in enumerate(contexts):
            if c.req.sampling_params.ignore_eos:
                continue
            stop_ids = set(c.req.sampling_params.stop_token_ids or ()) | set(c.req.eos_token_ids or ())
            tokenizer = c.req.tokenizer
            if tokenizer.eos_token_id is not None:
                stop_ids.add(tokenizer.eos_token_id)
            stop_ids.update(getattr(tokenizer, "additional_stop_token_ids", ()) or ())
            if stop_ids:
                stops = torch.tensor(sorted(stop_ids), device=self.device)
                stop_at = torch.where((predictions[i, :, None] == stops).any(-1), positions[0] + 1, self.width).amin()
                emitted[i] = torch.minimum(emitted[i], stop_at)
        accepted = torch.minimum(first_mismatch, emitted)
        packet = torch.stack((emitted, accepted), dim=-1).tolist()
        return predictions, packet

    @span("das_wave.commit")
    def _commit(self, batches, contexts, widths, slots, predictions, packet, on_publish):
        outputs, retained = [], []
        for i, (batch, c, width, (count, accepted)) in enumerate(zip(batches, contexts, widths, packet, strict=True)):
            kept = slots[i, :count]
            self.req_to_token_pool.req_to_token[c.row, c.length:c.length+count] = kept.to(torch.int32)
            retained.append(kept)
            c.length += count
            c.emitted += count
            c.anchor = predictions[i, count-1:count].clone()
            c.req.kv.kv_allocated_len = c.length
            c.cycles += 1
            c.proposed += width
            c.tested += count
            c.accepted += accepted
            lengths = torch.full_like(batch.seq_lens, c.length)
            ids = predictions[i, :count].clone()
            result = GenerationBatchResult(
                logits_output=LogitsProcessorOutput(next_token_logits=None),
                next_token_ids=ids, can_run_cuda_graph=False,
                accept_lens=torch.full_like(lengths, count, dtype=torch.int32),
                new_seq_lens=lengths, next_draft_input=DASWaveInput(c.anchor, lengths),
                speculative_num_draft_tokens=self.speculative_num_draft_tokens,
                speculative_output_stride=count, num_non_draft_tokens_per_req=count-accepted)
            if on_publish is not None:
                on_publish(batch, lengths)
            outputs.append((batch, result))
        return outputs, torch.cat(retained)

    @torch.no_grad()
    @span("das_wave.cycle")
    def forward_recurrent(self, batches, on_publish=None):
        batches = [b for b in batches if b.reqs[0].rid in self.contexts
                   and self.contexts[b.reqs[0].rid].emitted < b.reqs[0].sampling_params.max_new_tokens]
        if not batches:
            return []
        contexts = [self.contexts[b.reqs[0].rid] for b in batches]
        widths = [min(self.width, c.req.sampling_params.max_new_tokens - c.emitted) for c in contexts]
        # Reserve bounded scratch once. Target and draft never share logical IDs.
        allocated = self.token_to_kv_pool_allocator.alloc(2 * len(contexts) * self.width)
        if allocated is None:
            raise RuntimeError("DAS-Wave allocation reserve exhausted")
        self.temporary = allocated
        draft, target = allocated.reshape(2, len(contexts), self.width).unbind(0)
        retained = None
        try:
            with device_timer_ctx(self.model_runner.device_timer, "decode"):
                guesses = self._draft(contexts, widths, draft)
                logits, owners, columns = self._verify(contexts, widths, target, guesses)
                predictions, packet = self._decision(contexts, widths, guesses, logits, owners, columns)
                outputs, retained = self._commit(batches, contexts, widths, target, predictions, packet, on_publish)
            return outputs
        finally:
            free = allocated if retained is None else allocated[~torch.isin(allocated, retained)]
            self.token_to_kv_pool_allocator.free(free)
            self.temporary = None

    def forward_batch_generation(self, batch, on_publish=None, pp_proxy_tensors=None):
        if batch.forward_mode.is_extend():
            raise RuntimeError("DAS-Wave prefill must use native recurrent executor")
        callback = (lambda _b, lengths: on_publish(lengths)) if on_publish else None
        return GenerationBatchResult(recurrent_outputs=self.forward_recurrent(split_batch(batch), callback))

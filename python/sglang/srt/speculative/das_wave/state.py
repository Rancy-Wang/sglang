"""Pure greedy verification oracle for lifecycle tests."""


def verify_prefix(proposals, predictions, remaining, stop_ids=()):
    if len(proposals) != len(predictions) or not proposals or remaining < 1:
        raise ValueError("Verification requires matching nonempty rows and a positive budget")
    emitted, accepted, tested = [], 0, 0
    for draft, target in zip(proposals[:remaining], predictions[:remaining]):
        tested += 1
        match = draft == target
        accepted += int(match)
        emitted.append(target)
        if not match or target in stop_ids:
            break
    return emitted, accepted, tested

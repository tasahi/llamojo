""" Unit & parity test for Timestral Categorical Covariate Target Encoding (`tests/test_timestral_categorical.mojo`)."""

from std.math import abs, isnan
from std.memory.alloc import unsafe_alloc
from std.memory import unsafe_memset_zero

from llamojo.backend.kernels_quant import FloatPtr
from llamojo.timestral import target_encode, TargetEncodingResult


def main() raises:
    print("=== Running Timestral Categorical Covariate Target Encoding Test ===")

    # Target: [10.0, 20.0, NaN, 40.0, 50.0]
    # Past codes: [1.0, 2.0, 1.0, NaN, 2.0]
    # Future codes: [1.0, NaN, 99.0, 2.0]
    #
    # Upstream python timestral.covariate_encoding.target_encode output:
    # past_enc: [20.0, 33.333332, 20.0, 35.0, 33.333332]
    # fut_enc:  [20.0, 35.0, 30.0, 33.333332]
    #
    # Let's verify manually:
    # observed targets: 10, 20, 40, 50 -> count = 4, sum = 120 -> mean = 30.0
    # unique past finite codes: [1.0, 2.0] (categories: 0->1.0, 1->2.0)
    # nan_slot: 2
    # unseen: 3
    #
    # Category 1.0 (slot 0):
    #   past_codes has 1.0 at idx 0 (target=10) and idx 2 (target=NaN - omitted).
    #   sum = 10, count = 1
    #   encoded = (1.0 * 30.0 + 10) / (1.0 + 1) = 40 / 2 = 20.0
    # Category 2.0 (slot 1):
    #   past_codes has 2.0 at idx 1 (target=20) and idx 4 (target=50).
    #   sum = 70, count = 2
    #   encoded = (1.0 * 30.0 + 70) / (1.0 + 2) = 100 / 3 = 33.333333...
    # NaN slot (slot 2):
    #   past_codes has NaN at idx 3 (target=40).
    #   sum = 40, count = 1
    #   encoded = (1.0 * 30.0 + 40) / (1.0 + 1) = 70 / 2 = 35.0
    # Unseen slot (slot 3):
    #   encoded = 30.0
    #
    # past_enc:
    #   idx 0 (code 1.0) -> slot 0 -> 20.0
    #   idx 1 (code 2.0) -> slot 1 -> 33.333332
    #   idx 2 (code 1.0) -> slot 0 -> 20.0
    #   idx 3 (code NaN) -> slot 2 -> 35.0
    #   idx 4 (code 2.0) -> slot 1 -> 33.333332
    #
    # fut_enc:
    #   idx 0 (code 1.0) -> slot 0 -> 20.0
    #   idx 1 (code NaN) -> slot 2 -> 35.0
    #   idx 2 (code 99.0) -> unseen -> 30.0
    #   idx 3 (code 2.0) -> slot 1 -> 33.333332

    var t_len = 5
    var h_len = 4

    var target = unsafe_alloc[Float32](t_len)
    var past_codes = unsafe_alloc[Float32](t_len)
    var future_codes = unsafe_alloc[Float32](h_len)

    # NaN representation: 0.0 / 0.0
    var nan_val = Float32(0.0) / Float32(0.0)

    target[unsafe_offset=0] = 10.0
    target[unsafe_offset=1] = 20.0
    target[unsafe_offset=2] = nan_val
    target[unsafe_offset=3] = 40.0
    target[unsafe_offset=4] = 50.0

    past_codes[unsafe_offset=0] = 1.0
    past_codes[unsafe_offset=1] = 2.0
    past_codes[unsafe_offset=2] = 1.0
    past_codes[unsafe_offset=3] = nan_val
    past_codes[unsafe_offset=4] = 2.0

    future_codes[unsafe_offset=0] = 1.0
    future_codes[unsafe_offset=1] = nan_val
    future_codes[unsafe_offset=2] = 99.0
    future_codes[unsafe_offset=3] = 2.0

    var res = target_encode(
        target=target,
        past_codes=past_codes,
        t_len=t_len,
        future_codes=future_codes,
        h_len=h_len,
        smooth=1.0,
    )

    var expected_past = List[Float32]()
    expected_past.append(20.0)
    expected_past.append(33.333332)
    expected_past.append(20.0)
    expected_past.append(35.0)
    expected_past.append(33.333332)

    var expected_fut = List[Float32]()
    expected_fut.append(20.0)
    expected_fut.append(35.0)
    expected_fut.append(30.0)
    expected_fut.append(33.333332)

    print("Checking past encoding parity...")
    for i in range(t_len):
        var actual = res.past_encoded[unsafe_offset=i]
        var exp = expected_past[i]
        var diff = abs(actual - exp)
        print("  past[", i, "] actual=", actual, " expected=", exp, " diff=", diff)
        if diff > 1e-4:
            raise Error("Parity mismatch in past encoding!")

    print("Checking future encoding parity...")
    for i in range(h_len):
        var actual = res.future_encoded[unsafe_offset=i]
        var exp = expected_fut[i]
        var diff = abs(actual - exp)
        print("  fut[", i, "] actual=", actual, " expected=", exp, " diff=", diff)
        if diff > 1e-4:
            raise Error("Parity mismatch in future encoding!")

    print("Checking single-pass without future_codes...")
    var res_nofut = target_encode(
        target=target,
        past_codes=past_codes,
        t_len=t_len,
    )
    for i in range(t_len):
        var diff = abs(res_nofut.past_encoded[unsafe_offset=i] - expected_past[i])
        if diff > 1e-4:
            raise Error("Mismatch in past encoding without future!")
    res_nofut.free()

    target.unsafe_free()
    past_codes.unsafe_free()
    future_codes.unsafe_free()
    res.free()

    print("Categorical Covariate Target Encoding test passed successfully!")

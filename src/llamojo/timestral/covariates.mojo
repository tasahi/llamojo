""" Categorical Covariate Target Encoding (`src/llamojo/timestral/covariates.mojo`).

Implements Gap A remedy for the Timestral engine:
Matches upstream `timestral.covariate_encoding.target_encode`:

    (smooth * mean_i + sum_{i,c}) / (smooth + count_{i,c})

over the series context, with `smooth = 1.0` by default.
- Missing values (NaN) in past_codes form their own distinct category slot (`nan_slot`).
- Target entries that are non-finite (NaN / Inf) are excluded from means and category sums.
- Unseen categories in future_codes resolve to the overall series mean `mean_i`.
"""

from std.math import isnan, isinf
from std.memory import Pointer, unsafe_memset_zero, unsafe_memcpy
from std.memory.alloc import unsafe_alloc
from std.collections import List

from ..types import null_ptr
from ..backend.kernels_quant import FloatPtr


# ─── Target Encoding Result Container ────────────────────────────────────────

struct TargetEncodingResult(Movable):
    """ Container holding encoded past and optional future float arrays."""
    var past_encoded: FloatPtr
    var t_len: Int
    var future_encoded: FloatPtr
    var h_len: Int
    var has_future: Bool

    def __init__(
            out self,
            past_encoded: FloatPtr,
            t_len: Int,
            future_encoded: FloatPtr = null_ptr[Float32](),
            h_len: Int = 0,
            has_future: Bool = False,
        ):
        self.past_encoded = past_encoded
        self.t_len = t_len
        self.future_encoded = future_encoded
        self.h_len = h_len
        self.has_future = has_future

    def free(mut self):
        if Int(self.past_encoded) != 0:
            self.past_encoded.unsafe_free()
            self.past_encoded = null_ptr[Float32]()
        if self.has_future and Int(self.future_encoded) != 0:
            self.future_encoded.unsafe_free()
            self.future_encoded = null_ptr[Float32]()


# ─── Pure Mojo Target Encoding ───────────────────────────────────────────────

def target_encode(
        target: FloatPtr,
        past_codes: FloatPtr,
        t_len: Int,
        future_codes: FloatPtr = null_ptr[Float32](),
        h_len: Int = 0,
        smooth: Float32 = 1.0,
    ) -> TargetEncodingResult:
    """ Computes per-series smoothed mean-target encoding matching upstream Timestral.

    Args:
        target: (t_len,) context target values. Non-finite values are omitted.
        past_codes: (t_len,) categorical codes (floats); NaN represents missing code.
        t_len: Number of historical time steps.
        future_codes: Optional (h_len,) codes over future horizon.
        h_len: Horizon length (0 if none).
        smooth: Smoothing weight toward item mean (default 1.0).

    Returns:
        TargetEncodingResult with past_encoded and future_encoded arrays.
    """
    # 1. Compute valid observed item mean
    var target_sum: Float64 = 0.0
    var item_count: Float64 = 0.0
    for i in range(t_len):
        var y = target[unsafe_offset=i]
        if not isnan(y) and not isinf(y):
            target_sum += Float64(y)
            item_count += 1.0

    var item_mean: Float32 = 0.0
    if item_count > 0.0:
        item_mean = Float32(target_sum / item_count)

    # 2. Extract unique finite categories from past_codes (insertion-sorted)
    var categories = List[Float32]()
    for i in range(t_len):
        var c = past_codes[unsafe_offset=i]
        if not isnan(c) and not isinf(c):
            # Check if exists
            var exists = False
            for j in range(len(categories)):
                if categories[j] == c:
                    exists = True
                    break
            if not exists:
                categories.append(c)

    # Sort categories ascending
    var n_cat = len(categories)
    for i in range(n_cat):
        for j in range(i + 1, n_cat):
            if categories[j] < categories[i]:
                var tmp = categories[i]
                categories[i] = categories[j]
                categories[j] = tmp

    # nan_slot is assigned to index n_cat
    var nan_slot = n_cat
    # unseen slot is assigned to index n_cat + 1
    var unseen_slot = n_cat + 1
    var num_slots = n_cat + 2

    # 3. Accumulate weighted sums and counts for slots 0..nan_slot
    var sums = unsafe_alloc[Float64](num_slots)
    var counts = unsafe_alloc[Float64](num_slots)
    unsafe_memset_zero(sums, num_slots)
    unsafe_memset_zero(counts, num_slots)

    # Determine slots for past_codes
    var past_slots = unsafe_alloc[Int](t_len)
    for i in range(t_len):
        var c = past_codes[unsafe_offset=i]
        if isnan(c) or isinf(c):
            past_slots[unsafe_offset=i] = nan_slot
        else:
            var found = False
            for cat_idx in range(n_cat):
                if categories[cat_idx] == c:
                    past_slots[unsafe_offset=i] = cat_idx
                    found = True
                    break
            if not found:
                past_slots[unsafe_offset=i] = unseen_slot

        # Accumulate if target is finite
        var y = target[unsafe_offset=i]
        if not isnan(y) and not isinf(y):
            var slot = past_slots[unsafe_offset=i]
            sums[unsafe_offset=slot] = sums[unsafe_offset=slot] + Float64(y)
            counts[unsafe_offset=slot] = counts[unsafe_offset=slot] + 1.0

    # 4. Construct lookup table
    # lookup[s] = (smooth * item_mean + sum[s]) / (smooth + count[s])
    # lookup[unseen_slot] = item_mean
    var lookup = unsafe_alloc[Float32](num_slots)
    for s in range(nan_slot + 1):
        var s_val = (Float64(smooth) * Float64(item_mean) + sums[unsafe_offset=s]) / (Float64(smooth) + counts[unsafe_offset=s])
        lookup[unsafe_offset=s] = Float32(s_val)
    lookup[unsafe_offset=unseen_slot] = item_mean

    sums.unsafe_free()
    counts.unsafe_free()

    # 5. Populate past_encoded
    var past_encoded = unsafe_alloc[Float32](t_len)
    for i in range(t_len):
        var slot = past_slots[unsafe_offset=i]
        past_encoded[unsafe_offset=i] = lookup[unsafe_offset=slot]
    past_slots.unsafe_free()

    # 6. Populate future_encoded if requested
    var future_encoded = null_ptr[Float32]()
    var has_future = False
    if Int(future_codes) != 0 and h_len > 0:
        has_future = True
        future_encoded = unsafe_alloc[Float32](h_len)
        for i in range(h_len):
            var c = future_codes[unsafe_offset=i]
            var slot = unseen_slot
            if isnan(c) or isinf(c):
                slot = nan_slot
            else:
                for cat_idx in range(n_cat):
                    if categories[cat_idx] == c:
                        slot = cat_idx
                        break
            future_encoded[unsafe_offset=i] = lookup[unsafe_offset=slot]

    lookup.unsafe_free()

    return TargetEncodingResult(
        past_encoded=past_encoded,
        t_len=t_len,
        future_encoded=future_encoded,
        h_len=h_len,
        has_future=has_future,
    )

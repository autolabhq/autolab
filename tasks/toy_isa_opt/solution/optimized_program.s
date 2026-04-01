; Optimized dot product — 4× unrolled with multiple accumulators
; Technique: loop unrolling + instruction scheduling to hide load/mul latency
;
; 4 independent multiply chains → 4 accumulators (r1, r9, r10, r11)
; Loads for next pair issued while previous mul is in flight
; Final result: r1 = r1 + r9 + r10 + r11
;
; Registers:
;   r0  = 0 (always zero)
;   r1  = accumulator 0  (also final result)
;   r2  = loop counter (counts by 4, up to 512)
;   r3  = limit = 512
;   r4  = ptr_A
;   r5  = ptr_B
;   r6..r8   = temps for element 0
;   r12..r14 = temps for element 1
;   r9  = accumulator 1
;   r10 = accumulator 2  (temps r6,r7 reused)
;   r11 = accumulator 3
;   r15 = scratch for element 2/3 products

    addi r1,  r0, 0      ; acc0 = 0
    addi r9,  r0, 0      ; acc1 = 0
    addi r10, r0, 0      ; acc2 = 0
    addi r11, r0, 0      ; acc3 = 0
    addi r2,  r0, 0      ; i = 0
    addi r3,  r0, 512    ; N = 512
    addi r4,  r0, 0      ; ptr_A = 0
    addi r5,  r0, 512    ; ptr_B = 512

loop:
    ; --- Element 0: A[i], B[i] ---
    ld   r6,  0(r4)      ; r6  = A[i+0]  [5-cycle load]
    ld   r7,  0(r5)      ; r7  = B[i+0]  [5-cycle load]

    ; --- Element 1: A[i+1], B[i+1] --- (issued while loads 0 complete)
    ld   r12, 1(r4)      ; r12 = A[i+1]  [5-cycle load]
    ld   r13, 1(r5)      ; r13 = B[i+1]  [5-cycle load]

    ; --- Element 2: A[i+2], B[i+2] ---
    ld   r14, 2(r4)      ; r14 = A[i+2]  [5-cycle load]
    ld   r15, 2(r5)      ; r15 = B[i+2]  [5-cycle load]

    ; --- Multiply element 0 (stalls until r6,r7 ready) ---
    mul  r8,  r6,  r7    ; r8 = A[i]*B[i]

    ; --- Load element 3 while mul 0 is in flight ---
    ld   r6,  3(r4)      ; r6  = A[i+3]  (reuse r6)
    ld   r7,  3(r5)      ; r7  = B[i+3]  (reuse r7)

    ; --- Multiply elements 1,2 (may stall for their loads) ---
    mul  r12, r12, r13   ; r12 = A[i+1]*B[i+1]
    mul  r14, r14, r15   ; r14 = A[i+2]*B[i+2]

    ; --- Accumulate element 0 (stalls until r8 ready) ---
    add  r1,  r1,  r8    ; acc0 += A[i]*B[i]

    ; --- Multiply element 3 (stalls until r6,r7 ready) ---
    mul  r15, r6,  r7    ; r15 = A[i+3]*B[i+3]

    ; --- Advance pointers and counter while muls complete ---
    addi r4,  r4,  4     ; ptr_A += 4
    addi r5,  r5,  4     ; ptr_B += 4
    addi r2,  r2,  4     ; i += 4

    ; --- Accumulate elements 1,2,3 (stall for their muls) ---
    add  r9,  r9,  r12   ; acc1 += A[i+1]*B[i+1]
    add  r10, r10, r14   ; acc2 += A[i+2]*B[i+2]
    add  r11, r11, r15   ; acc3 += A[i+3]*B[i+3]

    bne  r2,  r3,  loop  ; if i != 512, loop  [+2 cycle flush when taken]

    ; --- Final reduction ---
    add  r1, r1,  r9     ; r1 = acc0 + acc1
    add  r1, r1,  r10    ; r1 += acc2
    add  r1, r1,  r11    ; r1 += acc3
    halt

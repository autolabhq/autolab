; Naive dot product — maximum stalls, no scheduling, single accumulator
; Kernel: sum = A[0]*B[0] + A[1]*B[1] + ... + A[511]*B[511]
;
; Memory layout (word-addressed):
;   A[0..511]   at addresses   0..511
;   B[0..511]   at addresses 512..1023
;
; Registers:
;   r0 = 0 (always zero)
;   r1 = sum  (result — must be in r1 at halt)
;   r2 = loop counter i
;   r3 = N = 512
;   r4 = ptr_A (current address of A[i])
;   r5 = ptr_B (current address of B[i])
;   r6 = tmp: A[i]
;   r7 = tmp: B[i]
;   r8 = product: A[i] * B[i]

    addi r2, r0, 0      ; i = 0
    addi r3, r0, 512    ; N = 512
    addi r4, r0, 0      ; ptr_A = base of A
    addi r5, r0, 512    ; ptr_B = base of B
    addi r1, r0, 0      ; sum = 0

loop:
    ld   r6, 0(r4)      ; r6 = A[i]         [5-cycle load]
    ld   r7, 0(r5)      ; r7 = B[i]         [5-cycle load, stalls for r6 not needed but r7 not ready]
    mul  r8, r6, r7     ; r8 = A[i]*B[i]    [5-cycle mul, stalls waiting for r6 and r7]
    add  r1, r1, r8     ; sum += product     [stalls waiting for r8]
    addi r4, r4, 1      ; ptr_A++
    addi r5, r5, 1      ; ptr_B++
    addi r2, r2, 1      ; i++
    bne  r2, r3, loop   ; branch back if i != N  [+2 cycle flush when taken]
    halt

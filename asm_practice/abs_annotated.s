.section __DATA,__data
test_value: .double -12.5

.section __TEXT,__text
.globl _main
.p2align 2
_main:
    // Load test value
    adrp x9, test_value@PAGE
    add x9, x9, test_value@PAGEOFF
    ldr d0, [x9]    // d0 = -12.5
    
    // Implement absolute value
    fmov d1, #0.0   // Load zero for comparison
    fcmp d0, d1     // Compare with zero
    b.ge .positive  // Branch if greater or equal to zero -- conditional ops use the last cmp flag
    
    // Negative case: multiply by -1
    fneg d0, d0     // d0 = -d0
    
.positive:
    // d0 now contains absolute value
    
    // Convert to integer for return
    fcvtzs w0, d0
    ret
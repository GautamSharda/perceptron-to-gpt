.section __DATA,__data
value_a: .double 10.0
value_b: .double 5.0

.section __TEXT,__text
.globl _main
.p2align 2
_main:
    // Load values
    adrp x9, value_a@PAGE
    add x9, x9, value_a@PAGEOFF
    ldr d0, [x9]    // d0 = 10.0
    
    adrp x9, value_b@PAGE
    add x9, x9, value_b@PAGEOFF
    ldr d1, [x9]    // d1 = 5.0
    
    // Calculate: (a + b) * (a - b)
    fadd d2, d0, d1  // d2 = a + b float addition is fadd instead of add
    fsub d3, d0, d1  // d3 = a - b
    fmul d4, d2, d3  // d4 = (a + b) * (a - b)
    
    // Convert to integer for return
    fcvtzs w0, d4
    ret
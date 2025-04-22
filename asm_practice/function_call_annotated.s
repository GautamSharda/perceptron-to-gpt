.section __DATA,__data
test_value: .double -7.5

.section __TEXT,__text
.globl _main
.extern _fabs // cool, think this tells linker to import this std lib function
.p2align 2
_main:
    // meta: stack pointer tells you the stack ends, frame pointer tells you where data related to your function begins, and is usually followed by the current function's return address. 
    // Save link register and frame pointer
    stp x29, x30, [sp, #-16]! // store PAIR: x29 (frame pointer) and x30 (return address) at sp - 16 -- note ! means that deference is computed first AND WRITTEN BACK (SO THE SP IS ACTUALLY MOVED 16 LOWER: sp = sp 16, then we store at sp)
    // so basically the above gives:
    // [sp]   = x29  -- Store old frame pointer
    // [sp+8] = x30  -- Store main return address
    mov x29, sp
    
    // Load test value
    adrp x9, test_value@PAGE
    add x9, x9, test_value@PAGEOFF
    ldr d0, [x9]    // d0 = -7.5 (argument for fabs)
    
    // Call fabs function
    bl _fabs        // Result will be in d0
    
    // Convert result to integer for return
    fcvtzs w0, d0
    
    // Restore registers and return
    ldp x29, x30, [sp], #16
    ret
.section __DATA,__data // conevntion to store mutable numerical data in DATA SEGMENT in data section
my_constant: .double 3.14 // note the colon syntax -- var: .type val

.section __TEXT,__text
.globl _main
.p2align 2
_main: // implicitly return typed to int btw.
    // Load constant from memory
    adrp x9, my_constant@PAGE // Memory is divded into pages (4k or 16 kb chunks). Programs don't know their location in memory -- this gives you the page number of the constant.
    add x9, x9, my_constant@PAGEOFF // This gives you the offset within the page of your constant, which we add to the page address to get the constant address
    ldr d0, [x9]    // d0 now contains 3.14. We dereference the memory address of the constant to get the value and we load it into register d0. 0 usually indicates return val (tho not here) and d is double.
    
    // Convert to integer for return
    fcvtzs w0, d0 // fcvtzs = (floating point) (convert to) (zero) (signed integer) -- which means -3.9 is -3.0 and 3.9 is 3: they round towards to 0. Put in w0, which is integer return.
    ret
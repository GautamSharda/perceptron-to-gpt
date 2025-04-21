.section __TEXT, __text // required because it tells the assembler to put the following stuff in the place where executable code should be
// __TEXT is a SEGMENT and __text is a SECTION *within* a SEGMENT.
// The heirarchy is:
// Executable File
// └── __TEXT (Segment - read-only, executable)
//     ├── __text (Section - actual code)
//     ├── __cstring (Section - string constants)
//     └── __const (Section - other constants)
.globl _main // required because it tells the assembler to make this symbol global and the OS uses main to know ehre to begin execution
.p2align 2 // required because ARM64 instructions are always 4 bytes long and must be 4-byte aligned. The processor expects to find instructions at addresses divisible by 4. This tells the assembler to get to an address divisible by 4 before putting the next instruction (perhaps by using some padding).
_main: // signify start of the main function
    mov w0, #0 // register 0 holds return value of a funtion in ARM64 and w indicates integer.
    ret
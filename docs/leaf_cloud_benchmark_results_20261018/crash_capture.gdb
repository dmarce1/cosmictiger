set pagination off
set confirm off
set print thread-events off
handle SIGSEGV stop print nopass
run
info registers rax rbp rip
x/gx $rbp-0x9e8
info address nodes
x/gx &nodes
thread apply all bt 18
quit

本程序测试当数据在远程HBM上时，不作任何kernel优化，直接通过nvlink direct访问远程内存并作本地计算，其性能会下降多少。

编译：
make
请注意把Makefile中的-arch=sm_70改为对应GPU架构，如H系列GPU为sm_90a。

运行：
./run_tests.sh
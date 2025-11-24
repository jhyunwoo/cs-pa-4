GPTMini Implementation and Optimization
CAS2107-01: Computer System,
Programming Assignment #4
(due on Friday, Dec 5, 2025 by 11:59 pm)
Objectives
This assignment aims to provide hands-on experience with performance-critical kernel optimization in
modern machine learning systems. Through implementing and optimizing the matrix multiplication
kernel used in GPTMini, you will:
1. Implement a correct, general-purpose matrix multiplication function.
2. Apply system-level optimization techniques such as:
○
loop reordering
○
cache-aware blocking
○
SIMD vectorization (AVX / FMA)
By completing the assignment, you will see how low-level optimizations translate into real
improvements in deep learning workloads.
Background
GPTMini is a neural network model that generates text. In simple terms, it works as follows: first, it
converts input text into numerical vectors through embedding; second, it processes information
through multiple Transformer blocks; and third, it predicts the next word as the final output.
Transformer models, including GPT-style architectures, perform extensive dense linear algebra
operations. Among these, matrix multiplication is the most computationally expensive and is used
throughout the network:
●
●
●
●
Q, K, V projections in Self-Attention
Attention score computation
FeedForward layers
Final output projection
In production systems, highly optimized BLAS libraries (OpenBLAS, MKL, cuBLAS) implement
sophisticated techniques such as cache tiling, SIMD vectorization, and fused multiply–add
instructions to accelerate these operations.
In this assignment, you will implement and optimize your own matrix multiplication kernel. GPTMini
is intentionally simplified so that the performance impact of your implementation is clearly visible.
Your optimized matrix
matrix
_
_
multiply() function will directly influence the runtime of the entire
GPTMini model.
1/4
Overview
All components of GPTMini are already implemented and provided. The naive implementation of the
matrix
matrix
_
_
multiply function is as follows.
float* matrix_matrix_multiply(const float* A, int m, int k, const float* B, int n) {
float* C = new float[m * n];
for (int i = 0; i < m; i++) {
for (int j = 0; j < n; j++) {
C[i * n + j] = 0.0f;
for (int p = 0; p < k; p++) {
C[i * n + j] += A[i * k + p] * B[p * n + j];
}
}
}
return C;
}
This function performs a matrix multiplication:
●
●
●
Input A: (m × k)
Input B: (k × n)
Output C: (m × n)
Your task is to optimize it for performance. This single kernel is used throughout the model, so
improving its speed benefits all Transformer layers.
Code Structure
GPTMini consists of the following components:
●
●
●
●
●
●
Embedding layer
Transformer blocks
Self-Attention (Q, K, V, Wo)
FeedForward (fc1, fc2)
LayerNorm (ln1, ln2)
LM Head
All linear layers eventually call matrix
matrix
_
_
multiply(). Thus, this function is the core
computation that determines overall inference speed.
Handout Instructions
Use the following command to retrieve the contents of this assignment.
$ cd ~
$ tar xvf /root/cs
_pa4.tar
2/4
Inside the cs
_pa4 directory, you will find two .cpp files, two .h files, and a Makefile. You will only
need to modify only one file: gpt
_
mini.cpp! If you also want to modify the Makefile, please leave a
comment in your submission report.
The values to be used as weights for GPTMini are stored under /tmp/cs
_pa4/weights on the course
server. When GPTMini runs, main.cpp loads all the weights from that location into memory and
passes them as pointers to the designated functions. In addition, the correct final layer output of
GPTMini for the given weights can be found in /tmp/cs
_pa4/answer/layer
final.out.
_
Compilation and Execution
Build and run locally with:
$ make
$ make run
The executable prints allocation logs, each decoding step’s token, elapsed seconds, and throughput. In
both the assignment environment and the grading environment, the configuration of GPTMini is as
follows: (vocabulary size, dmodel, dff, number of layer) = (4096, 2048, 2560, 1).
If you want to check whether your implementation is producing the correct result, use the following
command:
$ make run verbose
This command generates an output/layer
_
final.out file under the current working directory,
storing the final layer output of your GPTMini. It then compares this output with the correct answer,
and if the total error is less than 1, it prints ‘VALID’; otherwise, it prints ‘INVALID’
. You must receive
‘VALID’ in order to get credit.
Grading Policy
The baseline performance is around 0.6 tokens per second on our course server. The following table
shows the points you will earn depending on the achieved tokens/s of your program. Please note that
if the sum of errors in the final layer output of your program exceeds 1.0, it will not be reflected in the
evaluation.
Tokens/s achieved Points
Report +10 points
~ 0.6 0
0.6 ~ 30
2.0 ~ 60
3.0 ~ 80
5.0 ~ 90
3/4
In addition, we will open an online scoreboard. You can upload and test your gpt
_
mini.cpp file on
the scoreboard. The scoreboard also shows the Top-20 fastest cases with their tokens/s results. You
can compare your performance with top runners and plan your optimization efforts. The scoreboard
can be accessed at the following link. Please upload your gpt
_
mini.cpp with your student ID.
https://165.132.118.201:3300/
Scoreboard position Extra points
Top 10 +30 extra points
Top 3 Grade boost (ex. B- → B0 or B0 → B+, B+ → A-)
There is a 10% penalty per day after the submission deadline. You earn no points after 5 days of the
set
_
_
deadline.
Important Points
●
●
●
●
●
You should not use accelerators like CUDA. Your implementation will be tested in our course
server equipped with Intel(R) Xeon(R) Gold 5218 CPU, which supports AVX2/AVX-512
(SIMD) with FMA instruction.
○
FMA instruction: https://en.wikipedia.org/wiki/FMA
instruction
_
Use performance optimization techniques learned in the lecture.
Additional materials for SIMD and matrix multiplication optimization
○
SIMD:
https://drive.google.com/file/d/1P39Ygl54aNeZ2ykx
_
RVVuijSar7c1h0z/view?usp=dr
ive
link
■ AVX2 Intrinsics:
https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#a
vxnewtechs=A VX
○
Matrix multiplication
■
https://drive.google.com/file/d/1P36MMpcGPjDvmY7tC2wwVXAxz-S7PkIc
/view?usp=drive
link
_
https://drive.google.com/file/d/1P2ytDq7Mc
rN9cres7n8b8Nhvb01rJA
_
_
w?usp=drive
link
_
When in doubt, understand the naïve implementation.
This programming assignment is limited to a single thread. Don’t use multi threads or
multiprocesses.
Hand-in Instructions
Upload your gpt
_
mini.cpp and a PDF report describing your implementations and optimizations to
LearnUs.
■
/vie
4/4
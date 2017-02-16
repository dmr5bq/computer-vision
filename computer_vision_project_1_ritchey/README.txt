
Dominic Ritchey - dmr5bq
16 February 2017
CS 4501 - Prof. Barnes

Notes on execution:

In order to run the edge detector, from the command line enter:
    python canny.py

In order to run the corner detector, from the command line enter:
    python harris.py

Output images are saved into the results directory under the project root after execution.

Images are saved in the 'results' directory with names of the form:
    harris_result?threshold=<threshold>_<filename>.jpg
    canny_result?hthreshold=<number>&lthreshold=<number>_<filename>.jpg

NOTE: despite my efforts to optimize as much as possible, these programs are still taking several minutes to execute.
Bear with the execution, because both WILL terminate, it just takes a little while.
def selection_sort(A):
    print(A)
    n = len(A)

    # Pseudocode of SELECTION_SORT(A, n):
    # for i ← 1 to n do
    # 	min_i ← i
    # 	for j ← i + 1 to n do
    # 		if A[j] < A[min_i] do
    # 			min_i ← j
    # 	swap ← A[i]
    # 	A[i] ← A[min_i]
    # 	A[min_i] ← swap

    for i in range(n):
        min_i = i
        for j in range(i + 1, n):
            if A[j] < A[min_i]:
                min_i = j

        swap = A[i]
        A[i] = A[min_i]
        A[min_i] = swap

        print(A)


if __name__ == "__main__":
    A = [9, 1, 5, 8, 3, 2]
    selection_sort(A)

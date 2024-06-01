import numpy as np

def svd_program():
    # Meminta input matriks A dari pengguna
    n = int(input("Masukkan jumlah baris matriks A: "))
    m = int(input("Masukkan jumlah kolom matriks A: "))
    A = []
    print("Masukkan elemen-elemen matriks A:")
    for i in range(n):
        row = []
        for j in range(m):
            element = float(input("Masukkan elemen A[{}][{}]: ".format(i, j)))
            row.append(element)
        A.append(row)

    # Mencari SVD
    U, s, VT = np.linalg.svd(A)

    # Menampilkan hasil
    print("Matriks A:")
    print(A)

    print("Matriks U:")
    print(U)

    print("Matriks singular values (s):")
    print(s)

    print("Matriks VT:")
    print(VT)

def svd_complex_program():
    # Meminta input matriks A dari pengguna
    n = int(input("Masukkan jumlah baris matriks A: "))
    m = int(input("Masukkan jumlah kolom matriks A: "))
    A = []
    print("Masukkan elemen-elemen matriks A:")
    for i in range(n):
        row = []
        for j in range(m):
            element = complex(input("Masukkan elemen A[{}][{}]: ".format(i, j)))
            row.append(element)
        A.append(row)

    # Meminta input vektor B dari pengguna
    B = []
    print("Masukkan elemen-elemen vektor B:")
    for i in range(n):
        element = complex(input("Masukkan elemen B[{}][0]: ".format(i)))
        B.append(element)
    B = np.array(B)

    # Memecahkan persamaan menggunakan SVD
    U, s, V = np.linalg.svd(A, full_matrices=False)
    x = np.dot(V.T, np.dot(np.diag(1 / s), np.dot(U.T, B)))

    # Menampilkan solusi
    print("Solusi sistem persamaan linear:")
    for i, sol in enumerate(x):
        print(f"x{i+1} = {sol}")

def polynomial_characteristic_program():
    # Meminta input matriks dari pengguna
    n = int(input("Masukkan ukuran matriks (n x n): "))
    matrix = []
    print("Masukkan elemen-elemen matriks:")
    for i in range(n):
        row = []
        for j in range(n):
            element = float(input("Masukkan elemen M[{}][{}]: ".format(i, j)))
            row.append(element)
        matrix.append(row)
    matrix = np.array(matrix)

    def polynomial_characteristic(matrix):
        return np.poly(matrix)

    def eigen(matrix):
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        return eigenvalues, eigenvectors

    def find_diagonal_matrix(matrix):
        eigenvalues, eigenvectors = eigen(matrix)
        P = eigenvectors
        P_inv = np.linalg.inv(P)
        diagonal_matrix = np.diag(eigenvalues)
        transformed_matrix = np.dot(P_inv, np.dot(matrix, P))
        return P, diagonal_matrix, transformed_matrix

    # Hitung karakteristik polinomial
    characteristic = polynomial_characteristic(matrix)
    print("Karakteristik Polinomial:")
    print(characteristic)

    # Hitung eigenvalues dan eigenvectors
    eigenvalues, eigenvectors = eigen(matrix)
    print("Eigenvalues:")
    print(eigenvalues)
    print("Eigenvectors:")
    print(eigenvectors)

    # Cari matriks P untuk transformasi diagonalisasi
    P, diagonal_matrix, transformed_matrix = find_diagonal_matrix(matrix)
    print("Matriks P:")
    print(P)
    print("Matriks Diagonal:")
    print(diagonal_matrix)
    print("Transformasi Matriks:")
    print(transformed_matrix)

def linear_equation_numpy_program():
    # Meminta input matriks A dari pengguna
    n = int(input("Masukkan jumlah baris matriks A: "))
    m = int(input("Masukkan jumlah kolom matriks A: "))
    A = []
    print("Masukkan elemen-elemen matriks A:")
    for i in range(n):
        row = []
        for j in range(m):
            element = float(input("Masukkan elemen A[{}][{}]: ".format(i, j)))
            row.append(element)
        A.append(row)

    # Meminta input vektor B dari pengguna
    B = []
    print("Masukkan elemen-elemen vektor B:")
    for i in range(n):
        element = float(input("Masukkan elemen B[{}][0]: ".format(i)))
        B.append(element)
    B = np.array(B).reshape(n, 1)

    # Menggunakan fungsi solve dari numpy untuk menyelesaikan sistem persamaan
    # X adalah matriks kolom solusi
    X = np.linalg.solve(A, B)

    # Cetak solusi
    print("Solusi dari sistem persamaan:")
    for i, sol in enumerate(X):
        print("x{} = {}".format(i + 1, sol[0]))

def gauss_jordan_elimination(matrix, vector):
    n = len(matrix)

    # Menggabungkan matriks dan vektor menjadi augmented matrix
    augmented_matrix = [matrix[i] + [vector[i]] for i in range(n)]

    # Melakukan eliminasi Gauss-Jordan
    for i in range(n):
        # Pencarian baris dengan elemen terbesar di kolom i
        max_row = i
        for j in range(i + 1, n):
            if abs(augmented_matrix[j][i]) > abs(augmented_matrix[max_row][i]):
                max_row = j

        # Menukar baris teratas dengan baris dengan elemen terbesar
        augmented_matrix[i], augmented_matrix[max_row] = (
            augmented_matrix[max_row],
            augmented_matrix[i],
        )

        # Membuat elemen utama menjadi 1
        pivot = augmented_matrix[i][i]
        for j in range(i, n + 1):
            try:
                augmented_matrix[i][j] /= pivot
            except ZeroDivisionError:
                augmented_matrix[i][j] = 0

        # Mengeliminasi elemen-elemen di bawah dan di atas elemen utama
        for j in range(n):
            if j != i:
                factor = augmented_matrix[j][i]
                for k in range(i, n + 1):
                    augmented_matrix[j][k] -= factor * augmented_matrix[i][k]

    # Menghasilkan solusi
    solution = [augmented_matrix[i][n] for i in range(n)]
    return solution

def linear_equation_gauss_jordan_program():
    # Meminta input matriks dari pengguna
    n = int(input("Masukkan jumlah baris matriks: "))
    matrix = []
    print("Masukkan elemen-elemen matriks:")
    for i in range(n):
        row = []
        for j in range(n):
            element = float(input("Masukkan elemen M[{}][{}]: ".format(i, j)))
            row.append(element)
        matrix.append(row)

    # Meminta input vektor dari pengguna
    vector = []
    print("Masukkan elemen-elemen vektor:")
    for i in range(n):
        element = float(input("Masukkan elemen V[{}]: ".format(i)))
        vector.append(element)

    # Menyelesaikan sistem persamaan linier menggunakan metode eliminasi Gauss-Jordan
    solution = gauss_jordan_elimination(matrix, vector)

    # Menampilkan solusi
    print("Solusi:")
    for i in range(len(solution)):
        print("x{} = {}".format(i + 1, solution[i]))

def main():
    while True:
        print("Menu:")
        print("1. Program SVD")
        print("2. Program SVD Kompleks")
        print("3. Program Polinomial Karakteristik")
        print("4. Program Sistem Persamaan Linear (Numpy)")
        print("5. Program Sistem Persamaan Linear (Gauss-Jordan)")
        print("0. Keluar")

        choice = input("Masukkan pilihan menu: ")

        if choice == "1":
            print("=== Program SVD ===")
            svd_program()
        elif choice == "2":
            print("=== Program SVD Kompleks ===")
            svd_complex_program()
        elif choice == "3":
            print("=== Program Polinomial Karakteristik ===")
            polynomial_characteristic_program()
        elif choice == "4":
            print("=== Program Sistem Persamaan Linear (Numpy) ===")
            linear_equation_numpy_program()
        elif choice == "5":
            print("=== Program Sistem Persamaan Linear (Gauss-Jordan) ===")
            linear_equation_gauss_jordan_program()
        elif choice == "0":
            break
        else:
            print("Pilihan tidak valid. Silakan pilih menu yang sesuai.")

if __name__ == "__main__":
    main()

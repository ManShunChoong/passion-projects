from itertools import combinations

import numpy as np
import requests
from bs4 import BeautifulSoup

PUZZLE_URL = "https://menneske.no/sudoku/showpuzzle.html"
SOLUTION_URL = "https://menneske.no/sudoku/solution.html"

vectorize_len = np.vectorize(len)


def get_puzzles(puzzle_id=None):
    def get_grid(url):
        if puzzle_id:
            url += f"?number={puzzle_id}"

        response = requests.get(url=url)
        content = response.content

        soup = BeautifulSoup(content, "html.parser")

        return soup.find("div", {"class": "grid"})

    def get_values(div):
        problem = [
            [col.text.replace("\xa0", "123456789") for col in row.find_all("td")]
            for row in div.find_all("tr")
        ]
        return problem

    # Scrap puzzle
    grid_div = get_grid(url=PUZZLE_URL)
    puzzle = get_values(grid_div)

    # Scrap puzzle ID
    text = grid_div.find(text=True, recursive=False)
    digits = "".join(c for c in text if c.isdigit())
    puzzle_id = int(digits)

    # Scrap solution
    grid_div = get_grid(url=SOLUTION_URL)
    solution = get_values(grid_div)

    return puzzle_id, puzzle, solution


def get_test_puzzles():
    puzzle_id, solution = None, None
    puzzle = [
        [0, 0, 0, 2, 0, 0, 9, 0, 0],
        [0, 0, 0, 0, 0, 4, 6, 0, 0],
        [0, 0, 0, 0, 1, 3, 4, 8, 2],
        [0, 1, 0, 0, 0, 0, 8, 0, 0],
        [3, 8, 0, 0, 0, 0, 0, 1, 4],
        [0, 0, 2, 0, 0, 0, 0, 0, 0],
        [8, 6, 1, 5, 7, 0, 0, 0, 0],
        [0, 0, 5, 6, 0, 0, 0, 0, 0],
        [0, 0, 3, 0, 0, 8, 0, 0, 0],
    ]
    puzzle = [[str(cell).replace("0", "123456789") for cell in row] for row in puzzle]

    return puzzle_id, puzzle, solution


def display(puzzle, filled_only=False):
    for y, row in enumerate(puzzle):
        if y and not y % 3:
            print("+".join(["-" * 3 * 7] * 3))

        for x, cell in enumerate(row):
            cell = str(cell)
            if filled_only:
                cell = cell if cell in list("123456789") else "."
            else:
                cell = cell.replace("123456789", ".").replace("0", ".") if cell else "X"

            print("|" if x and not x % 3 else "", cell.center(7), sep="", end="")
        print()
    print()


def solve(puzzle):
    def available_values(y, x):
        row = puzzle[y]
        column = puzzle[:, x]
        y_start, x_start = y // 3 * 3, x // 3 * 3
        box = puzzle[y_start : y_start + 3, x_start : x_start + 3]

        return "".join(
            value
            for value in puzzle[y, x]
            if value not in row
            and value not in column
            and all(value not in row for row in box)
        )

    def hidden_singles(array):
        string = "".join(value for value in array[vectorize_len(array) > 1])
        values = array.ravel() if array.ndim > 1 else array

        return [
            value
            for value in set(string)
            if string.count(value) == 1 and value not in values
        ]

    def naked_pairs(array):
        values = array[vectorize_len(array) == 2]

        return [
            value for value in set(values) if np.count_nonzero(values == value) == 2
        ]

    def pointing_pairs(array):
        pairs = {}
        values = (
            value for value in "123456789" if all(value not in row for row in array)
        )

        for value in values:
            indices = [
                index for index, row in enumerate(array) if value in "".join(row)
            ]
            if len(indices) == 1:
                pairs[value] = indices[0]

        return pairs

    def claiming_pairs(array):
        pairs = {}
        values = (value for value in "123456789" if value not in array)

        for value in values:
            indices = [
                index
                for index in range(0, 9, 3)
                if any(value in cell for cell in array[index : index + 3])
            ]
            if len(indices) == 1:
                pairs[value] = indices[0]

        return pairs

    def naked_triples(array):
        values = array[(vectorize_len(array) == 2) | (vectorize_len(array) == 3)]

        if len(values) > 2:
            return [
                list(set("".join(triple)))
                for triple in combinations(values, 3)
                if len(set("".join(triple))) == 3
            ]

        return []

    def x_wings(array):
        result = {}

        for value in "123456789":
            x_indices, y_indices = np.where(
                (vectorize_len(array) > 1)
                & [[value in cell for cell in row] for row in array]
            )

            wings = []

            for x1, x2 in combinations(set(x_indices), 2):
                y1s = y_indices[x_indices == x1]
                y2s = y_indices[x_indices == x2]
                if len(y1s) == 2 and np.array_equal(y1s, y2s):
                    wings.append(([x1, x2], list(y1s)))

            if wings:
                result[value] = wings

        return result

    def hidden_pairs(array):
        pairs = {}

        for value1, value2 in combinations("123456789", 2):
            if array.ndim == 1:
                mask1 = [value1 in cell for cell in array]
                mask2 = [value2 in cell for cell in array]
            else:
                mask1 = [[value1 in cell for cell in row] for row in array]
                mask2 = [[value2 in cell for cell in row] for row in array]

            if np.array(mask1).sum() == 2 and mask1 == mask2:
                pairs[value1 + value2] = (
                    np.where(mask1)[0] if array.ndim == 1 else zip(*np.where(mask1))
                )

        return pairs

    def naked_quad(array):
        len_array = vectorize_len(array)

        if (len_array > 1).sum() > 6:
            filtered_array = array[(len_array > 1) & (len_array < 5)]

            for values in combinations(filtered_array, 4):
                if len(set("".join(values))) == 4:
                    return set("".join(values))

        return {}

    mode_counts = {
        "Naked single": 0,
        "Hidden single": 0,
        "Naked pair": 0,
        "Pointing pair": 0,
        "Claiming pair": 0,
        "Naked triple": 0,
        "X-wing": 0,
        "Hidden pair": 0,
        "Naked quad": 0,
    }

    puzzle = np.array(puzzle, dtype="U9")

    while True:
        while True:
            while True:
                while True:
                    while True:
                        while True:
                            while True:
                                while True:
                                    while True:
                                        last_puzzle = puzzle.copy()

                                        # Naked single
                                        for y, x in np.ndindex(puzzle.shape):
                                            if len(puzzle[y, x]) != 1:
                                                puzzle[y, x] = available_values(y, x)

                                        if np.array_equal(puzzle, last_puzzle):
                                            break

                                        mode_counts["Naked single"] += 1

                                    # Hidden single (box, row, column)
                                    for y_start in range(0, 9, 3):
                                        for x_start in range(0, 9, 3):
                                            box = puzzle[
                                                y_start : y_start + 3,
                                                x_start : x_start + 3,
                                            ]
                                            for value in hidden_singles(box):
                                                for y, x in np.ndindex(box.shape):
                                                    if value in box[y, x]:
                                                        box[y, x] = value

                                    for array in [puzzle, puzzle.T]:
                                        for row in array:
                                            for value in hidden_singles(row):
                                                for x, cell in enumerate(row):
                                                    if value in cell:
                                                        row[x] = value

                                    if np.array_equal(puzzle, last_puzzle):
                                        break

                                    mode_counts["Hidden single"] += 1

                                # Naked pair (box, row, column)
                                for y_start in range(0, 9, 3):
                                    for x_start in range(0, 9, 3):
                                        box = puzzle[
                                            y_start : y_start + 3, x_start : x_start + 3
                                        ]
                                        for pair in naked_pairs(box):
                                            for y, x in np.ndindex(box.shape):
                                                if (
                                                    len(box[y, x]) > 1
                                                    and any(
                                                        value in box[y, x]
                                                        for value in pair
                                                    )
                                                    and box[y, x] != pair
                                                ):
                                                    box[y, x] = (
                                                        box[y, x]
                                                        .replace(pair[0], "")
                                                        .replace(pair[1], "")
                                                    )

                                for array in [puzzle, puzzle.T]:
                                    for row in array:
                                        for pair in naked_pairs(row):
                                            for x, cell in enumerate(row):
                                                if (
                                                    len(cell) > 1
                                                    and any(
                                                        value in cell for value in pair
                                                    )
                                                    and cell != pair
                                                ):
                                                    row[x] = cell.replace(
                                                        pair[0], ""
                                                    ).replace(pair[1], "")

                                if np.array_equal(puzzle, last_puzzle):
                                    break

                                mode_counts["Naked pair"] += 1

                            # Pointing pair/triple (box)
                            for y_start in range(0, 9, 3):
                                for x_start in range(0, 9, 3):
                                    for array in [puzzle, puzzle.T]:
                                        box = array[
                                            y_start : y_start + 3, x_start : x_start + 3
                                        ]
                                        for value, sub_index in pointing_pairs(
                                            box
                                        ).items():
                                            row = array[y_start + sub_index]
                                            for x, cell in enumerate(row):
                                                if x // 3 * 3 != x_start:
                                                    row[x] = cell.replace(value, "")

                            if np.array_equal(puzzle, last_puzzle):
                                break

                            mode_counts["Pointing pair"] += 1

                        # Claiming pair/triple (row, column)
                        for array in [puzzle, puzzle.T]:
                            for index, row in enumerate(array):
                                for value, x_start in claiming_pairs(row).items():
                                    y_start = index // 3 * 3
                                    box = array[
                                        y_start : y_start + 3, x_start : x_start + 3
                                    ]
                                    for y, box_row in enumerate(box):
                                        if y != index % 3:
                                            for x, cell in enumerate(box_row):
                                                box_row[x] = cell.replace(value, "")

                        if np.array_equal(puzzle, last_puzzle):
                            break

                        mode_counts["Claiming pair"] += 1

                    # Naked triple (box, row, column)
                    for y_start in range(0, 9, 3):
                        for x_start in range(0, 9, 3):
                            box = puzzle[y_start : y_start + 3, x_start : x_start + 3]
                            for triple in naked_triples(box):
                                for y, x in np.ndindex(box.shape):
                                    if len(box[y, x]) > 1 and any(
                                        value not in triple for value in box[y, x]
                                    ):
                                        box[y, x] = (
                                            box[y, x]
                                            .replace(triple[0], "")
                                            .replace(triple[1], "")
                                            .replace(triple[2], "")
                                        )

                    for array in [puzzle, puzzle.T]:
                        for row in array:
                            for triple in naked_triples(row):
                                for x, cell in enumerate(row):
                                    if len(cell) > 1 and any(
                                        value not in triple for value in cell
                                    ):
                                        row[x] = (
                                            cell.replace(triple[0], "")
                                            .replace(triple[1], "")
                                            .replace(triple[2], "")
                                        )

                    if np.array_equal(puzzle, last_puzzle):
                        break

                    mode_counts["Naked triple"] += 1

                # X-wing (row, column)
                for array in [puzzle, puzzle.T]:
                    for value, indices_list in x_wings(array).items():
                        for (y_indices, x_indices) in indices_list:
                            # y_indices, x_indices = indices
                            for x in x_indices:
                                for y, cell in enumerate(array.T[x]):
                                    if len(cell) > 1 and y not in y_indices:
                                        array[y, x] = cell.replace(value, "")

                if np.array_equal(puzzle, last_puzzle):
                    break

                mode_counts["X-wing"] += 1

            # Hidden pair (box, row, column)
            for y_start in range(0, 9, 3):
                for x_start in range(0, 9, 3):
                    box = puzzle[y_start : y_start + 3, x_start : x_start + 3]
                    for value, indices in hidden_pairs(box).items():
                        for y, x in indices:
                            box[y, x] = value

            for array in [puzzle, puzzle.T]:
                for row in array:
                    for value, x_indices in hidden_pairs(row).items():
                        for x in x_indices:
                            row[x] = value

            if np.array_equal(puzzle, last_puzzle):
                break

            mode_counts["Hidden pair"] += 1

        # Naked quad (box, row, column)
        for y_start in range(0, 9, 3):
            for x_start in range(0, 9, 3):
                box = puzzle[y_start : y_start + 3, x_start : x_start + 3]
                quad = naked_quad(box)
                for y, x in np.ndindex(box.shape):
                    if len(box[y, x]) > 1 and any(
                        value not in quad for value in box[y, x]
                    ):
                        for value in quad:
                            box[y, x] = box[y, x].replace(value, "")

        for array in [puzzle, puzzle.T]:
            for row in array:
                quad = naked_quad(row)
                for x, cell in enumerate(row):
                    if len(cell) > 1 and any(value not in quad for value in cell):
                        for value in quad:
                            row[x] = row[x].replace(value, "")

        if np.array_equal(puzzle, last_puzzle):
            break

        mode_counts["Naked quad"] += 1

    return puzzle, mode_counts


if __name__ == "__main__":
    # Completed: 2707875
    # Stopped:
    # 4661539 | 75.0%
    # 3508471 | 42.1%
    # 505519  | 36.2%
    # 691776  | 12.5%

    puzzle_id, puzzle, solution = get_puzzles(691776)
    # puzzle_id, puzzle, solution = get_test_puzzles()
    display(puzzle)

    my_solution, mode_counts = solve(puzzle)
    # display(my_solution, filled_only=True)
    display(my_solution)

    print(f"Puzzle ID: {puzzle_id if puzzle_id else '?'}", end=" | ")
    inaccurate_mask = (my_solution != solution) & (vectorize_len(my_solution) == 1)

    if solution is not None and inaccurate_mask.any():
        print("Inaccurate")
    else:
        filled_count1 = (vectorize_len(puzzle) == 1).sum()
        empty_count1 = 81 - filled_count1
        empty_count2 = (vectorize_len(my_solution) == 1).sum() - filled_count1
        print(f"{empty_count2 / empty_count1:.1%} ({empty_count2}/{empty_count1})")
    print()

    for mode, count in mode_counts.items():
        print(f"{mode}:".rjust(14), str(count).rjust(2))

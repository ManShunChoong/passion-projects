"""
Bridge and Torch Problem

A + C <- 2A + B + C + D
2B    <- A + 3B + D

"""

from typing import List


def get_crossing_time(people: List[int]) -> int:
    if not people:
        return 0

    people = sorted(people)

    count = len(people)
    if count > 3:
        subtime = 2 * people[1]
        if subtime < people[0] + people[-2]:
            return people[0] + subtime + people[-1] + get_crossing_time(people[:-2])

    return people[0] * max(count - 2, 0) + sum(people[1:-1]) + people[-1]


def print_crossing_time(people: List[int]) -> None:
    print(f"People: {people}")
    time = get_crossing_time(people)
    print(f"Time: {time}")


def main() -> None:
    print_crossing_time(people=[2])
    print_crossing_time(people=[1, 8])
    print_crossing_time(people=[1, 2, 8])
    print_crossing_time(people=[1, 2, 5, 8])
    print_crossing_time(people=[1, 2, 5, 8, 9])
    print_crossing_time(people=[1, 4, 5, 8, 9, 12])


if __name__ == "__main__":
    main()

import math
import random
import sys
from abc import ABC, abstractmethod

import numpy as np
import pygame

BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)

COL_COUNT = 10
ROW_COUNT = 8
UNIT_SIZE = 80


class Drawable(ABC):
    @abstractmethod
    def draw(self):
        pass

    @staticmethod
    def draw_text(size, text, left, top):
        font = pygame.font.Font("freesansbold.ttf", size)
        text = font.render(text, True, WHITE)
        textRect = text.get_rect()
        textRect.topleft = (left, top)
        screen.blit(text, textRect)

    @staticmethod
    def draw_circle(colour, coord, radius):
        pygame.draw.circle(screen, colour, coord, radius)


class Board(Drawable):
    def __init__(self, col, row):
        self.col = col
        self.row = row
        self.space_arr = self.generate_map()
        self.char_arr = self.generate_char()
        print(self.char_arr.shape)

    def generate_map(self):
        def get_space_by_elem(col, row, elem, temp):
            if elem == 0:
                return Water(col, row, temp)
            elif elem == 1:
                return Soil(col, row, temp)
            elif elem == 9:
                return Volc(col, row, 200)

        size = (self.row, self.col)
        row_arr, col_arr = np.indices(size)

        temp_arr = np.random.randint(10, size=size).astype(float) + 25.5
        # temp_arr = np.zeros(size).astype(float)
        elem_arr = np.array(
            [
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 0, 0, 0, 0, 1, 1],
                [0, 1, 1, 1, 1, 0, 0, 0, 0, 1],
                [0, 1, 1, 1, 1, 0, 0, 0, 0, 1],
                [0, 0, 1, 1, 1, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 9, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
                # [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                # [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                # [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                # [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                # [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                # [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                # [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                # [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
            ]
        ).astype(int)

        vspace = np.vectorize(get_space_by_elem)
        return vspace(col_arr, row_arr, elem_arr, temp_arr).astype(Space)

    def generate_char(self):
        return np.array([Char(5, 1)]).astype(Char)

    def add_char(self, col, row):
        self.char_arr = np.append(self.char_arr, Char(col, row))

    def show(self):
        print("Temp:")
        vtemp = np.vectorize(lambda Space: Space.temp)
        print(vtemp(self.space_arr).astype(int))

    def draw(self):
        screen.fill(BLACK)

        # Top panel
        vtemp = np.vectorize(lambda Space: Space.temp)
        temp_arr = vtemp(self.space_arr)

        vheat = np.vectorize(lambda Space: Space.heat)
        heat_arr = vheat(self.space_arr)

        self.draw_text(32, "Farm", 40, 32)
        self.draw_text(20, "Temp: " + str(round(temp_arr.mean(), 2)), 200, 40)
        self.draw_text(20, "Heat: " + str(round(heat_arr.mean(), 2)), 360, 40)
        self.draw_text(20, str(int(pygame.time.get_ticks() / 1000)) + "s", 720, 40)

        # Grid cells
        vdraw = np.vectorize(Space.draw)
        vdraw(self.space_arr)

        # Characters
        vdraw = np.vectorize(Char.draw)
        vdraw(self.char_arr)

        pygame.display.update()

    def set_cell_temp(self, col, row, val=0):
        space = self.space_arr[row][col]
        space.temp = val

    def heat_up_cell(self, col, row, val=50):
        space = self.space_arr[row][col]
        space.heat_up()

    def cool_down(self, val=0.00001):
        vcool_down = np.vectorize(Space.cool_down)
        vcool_down(self.space_arr, val)

    def balance_temp(self, span=1.5, rate=0.02):
        def get_surrounding_mean(col, row):
            row_span, col_span = np.indices(self.space_arr.shape)

            vtemp = np.vectorize(lambda Space: Space.temp)
            temp_arr = vtemp(self.space_arr)
            surr_arr = (col_span - col) ** 2 + (row_span - row) ** 2 <= span**2

            return np.mean(temp_arr[surr_arr])

        if span >= 1 and rate > 0:
            row, col = np.indices(self.space_arr.shape)
            vget_surrounding_mean = np.vectorize(get_surrounding_mean)
            mean = vget_surrounding_mean(col, row)

            vbalance_temp = np.vectorize(Space.balance_temp)
            vbalance_temp(self.space_arr, mean)


class Space(Drawable):
    max_temp = 200

    def __init__(self, col, row, temp):
        self.col = col
        self.row = row
        self.temp = temp
        self.update_heat()

    @property
    @abstractmethod
    def heat_capa(self):
        pass

    def draw(self):
        temp_pct = int(self.temp) / self.max_temp
        temp_pct = round(temp_pct, 1)

        if self.heat_capa == 10:
            rgb = (255 * temp_pct, 255 * (1 - temp_pct), 0)
        else:
            rgb = (255 * temp_pct, 0, 255 * (1 - temp_pct))

        pygame.draw.rect(
            screen,
            rgb,
            (self.col * UNIT_SIZE, (self.row + 1) * UNIT_SIZE, UNIT_SIZE, UNIT_SIZE),
        )

        # Heat label
        font = pygame.font.Font("freesansbold.ttf", 12)
        text = font.render(str(int(self.temp)), True, WHITE)
        textRect = text.get_rect()
        textRect.center = ((self.col + 0.8) * UNIT_SIZE, (self.row + 1.65) * UNIT_SIZE)
        screen.blit(text, textRect)

        font = pygame.font.Font("freesansbold.ttf", 10)
        text = font.render(str(int(self.heat)), True, WHITE)
        textRect = text.get_rect()
        textRect.center = ((self.col + 0.8) * UNIT_SIZE, (self.row + 1.8) * UNIT_SIZE)
        screen.blit(text, textRect)

    def update_temp(self):
        self.temp = self.heat / self.heat_capa

    def update_heat(self):
        self.heat = self.temp * self.heat_capa

    def change_heat(self, val):
        self.heat += val
        self.update_temp()

        if self.temp > self.max_temp:
            self.temp = self.max_temp
            self.update_heat()
        elif self.temp < 0:
            self.temp = 0
            self.update_heat()

    def heat_up(self, val=50):
        if val > 0:
            self.change_heat(val)

    def cool_down(self, val=50):
        if val > 0:
            self.change_heat(-val)

    def balance_temp(self, mean, rate=0.01):
        val = (mean - self.temp) * rate
        self.change_heat(val)


class Volc(Space):
    heat_capa = 100

    def __init__(self, col, row, temp):
        super().__init__(col, row, temp)

    def change_heat(self, val):
        pass


class Air(Space):
    heat_capa = 1

    def __init__(self, col, row, temp):
        super().__init__(col, row, temp)


class Water(Space):
    heat_capa = 2

    def __init__(self, col, row, temp):
        super().__init__(col, row, temp)


class Soil(Space):
    heat_capa = 10

    def __init__(self, col, row, temp):
        super().__init__(col, row, temp)


class Rock(Space):
    heat_capa = 20

    def __init__(self, col, row, temp):
        super().__init__(col, row, temp)


class Char(Drawable):
    max_temp = 35
    min_temp = 25

    def __init__(self, col, row):
        self.col = col
        self.row = row

    def draw(self):
        coord = ((self.col + 0.5) * UNIT_SIZE, (self.row + 1.5) * UNIT_SIZE)
        self.draw_circle(YELLOW, coord, UNIT_SIZE / 5)

    def move(self, dir=None, step=1):
        step = int(step)

        if dir is None:
            pop = [0, 1, 2, 3]
            dir = random.choices(pop, [0.1, 0.1, 0.1, 0.1])[0]

        if step > 0:
            if dir == 0:  # UP
                if self.col > 0:
                    self.col -= step
            elif dir == 1:  # RIGHT
                if self.row < board.row - 1:
                    self.row += step
            elif dir == 2:  # DOWN
                if self.col < board.col - 1:
                    self.col += step
            elif dir == 3:  # LEFT
                if self.row > 0:
                    self.row -= step


board = Board(COL_COUNT, ROW_COUNT)
board.show()

# Initialise the pygame
pygame.init()

# Create the screen
width = COL_COUNT * UNIT_SIZE
height = (ROW_COUNT + 1) * UNIT_SIZE
size = (width, height)
screen = pygame.display.set_mode(size)

board.draw()
pygame.display.update()

# # Title and Icon
pygame.display.set_caption("Simulation")
# icon = pygame.image.load('ufo.png')
# pygame.display.set_icon(icon)

start_ticks = pygame.time.get_ticks()

MOVEEVENT, t, trail = pygame.USEREVENT + 1, 1000, []
pygame.time.set_timer(MOVEEVENT, t)

# Game Loop
is_pressed = False
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.MOUSEBUTTONDOWN:
            is_pressed = math.floor(event.pos[1] / UNIT_SIZE) > 0

        if event.type == pygame.MOUSEBUTTONUP:
            is_pressed = False

        if is_pressed:
            x, y = pygame.mouse.get_pos()
            col = math.floor(x / UNIT_SIZE)
            row = math.floor(y / UNIT_SIZE) - 1

            if row >= 0:
                board.heat_up_cell(col, row)

        if event.type == MOVEEVENT:
            vmove = np.frompyfunc(Char.move, 1, 1)
            vmove(board.char_arr)

    board.cool_down()
    board.balance_temp()

    board.draw()

    # screen.fill((255, 0, 0))

from collections import Counter

import numpy as np
import pandas as pd
import pulp as plp
from matplotlib import pyplot as plt
from scipy import integrate, optimize
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


# Simple Optimisation
def optimisation_problem1():
    # Let say there is a box, and we want to maximise the volume by changing
    # the length, width and height of the box.
    # Volume = Length (L) x Width (W) x Height (H)
    # Surface Area = (2 x L x W) + (2 x L x H) + (2 x W x H)
    # The surface area of the box is less than and equal to 10. (SA <= 10)
    # Tasks:
    # 1. Define the volume and surface area as a function
    # 2. Define the objective function and constraints function
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    def volume(box):
        length, width, height = box

        return length * width * height

    def surface_area(box):
        length, width, height = box

        return 2 * (length * width + length * height + width * height)

    # Objective function
    def objective(box):
        return -volume(box)

    # Constraint function
    def constraint(box):
        return 10 - surface_area(box)

    length0, width0, height0 = 1, 1, 1
    x0 = np.array([length0, width0, height0])

    # ineq -> f(x) >= 0 OR eq -> f(x) = 0
    constraints = {"type": "ineq", "fun": constraint}

    result = optimize.minimize(
        objective, x0, method="SLSQP", constraints=constraints, options={"disp": True}
    )
    print(result)

    optimal_x = result.x
    optimal_volume = -result.fun
    optimal_surface_area = surface_area(optimal_x)

    print("Optimal length:", optimal_x[0])
    print("Optimal width:", optimal_x[1])
    print("Optimal height:", optimal_x[2])
    print("Volume:", optimal_volume)
    print("Surface Area:", optimal_surface_area)


def optimisation_problem2():
    # Snowball Problem
    # How big a snowball should be after 30 seconds rolling down from a hill and to knock the tree down?
    # Let say it take 25000N force to knock down a tree.

    # Given parameters:
    k0 = 85
    c_d = 0.3
    g = 9.8
    rho = 350
    theta = np.radians(5)
    rho_a = 0.9
    beta = 0.07

    # Initial snowball conditions:
    mass0 = 10
    radius0 = (3 * mass0 / 4 / np.pi / rho) ** (1 / 3)
    velocity0 = 0
    position0 = 0

    # Formula for radius:
    def radius(mass):
        return (3 * mass / 4 / np.pi / rho) ** (1 / 3)

    # TODO: Function for dynamic snowball:
    def change_in_mass(time):
        return beta * k0 * np.exp(-beta * time)

    def change_in_radius(time, radius_):
        return beta * k0 * np.exp(-beta * time) / 4 / np.pi / rho / radius_**2

    def change_in_velocity(time, velocity, radius_):
        return (
            -15 * rho_a * c_d * velocity**2 / 56 / rho / radius_
            - 23
            * beta
            * k0
            * np.exp(-beta * time)
            * velocity
            / 28
            / np.pi
            / rho
            / radius_**3
            + 5 / 7 * g * np.sin(theta)
        )

    def change_in_position(velocity):
        return velocity

    def change_in_snowball(snowball, time):
        mass, radius_, velocity = snowball

        return [
            change_in_mass(time),
            change_in_radius(time, radius_),
            change_in_velocity(time, velocity, radius_),
        ]

    # Function for objective function:
    t = np.linspace(0, 30)
    critical_impact_force = 25000

    def kinetic_energy(mass, velocity):
        return mass * velocity**2 / 2

    def impact_force(mass, velocity, radius_):
        return kinetic_energy(mass, velocity) / radius_

    def objective(mass0_):
        radius0_ = radius(mass0_)
        y0 = [mass0_, radius0_, velocity0]
        solution = integrate.odeint(change_in_snowball, y0, t)
        mass, radius_, velocity = solution[-1]
        impact_force_ = impact_force(mass, velocity, radius_)

        return (impact_force_ - critical_impact_force) ** 2

    x0 = np.array(mass0)

    result = optimize.minimize(objective, x0, options={"disp": True})
    print(result)

    (optimal_mass0,) = result.x
    optimal_radius0 = radius(optimal_mass0)

    optimal_y0 = [optimal_mass0, optimal_radius0, velocity0]
    optimal_solution = integrate.odeint(change_in_snowball, optimal_y0, t)
    optimal_mass, optimal_radius, optimal_velocity = optimal_solution[-1]
    optimal_impact_force = impact_force(optimal_mass, optimal_velocity, optimal_radius)

    print("Optimal initial mass:", optimal_mass0)
    print("Optimal initial radius:", optimal_radius0)
    print("Optimal impact force:", optimal_impact_force)


# Linear Programming
def lp_problem1():
    # Create a LP problem
    prob = plp.LpProblem("LP_Problem_1")  # , plp.LpMaximize

    # Create variables
    x = plp.LpVariable("x", lowBound=0)
    y = plp.LpVariable("y", lowBound=0)

    # Add constraints
    prob += (2 * x + 3 * y >= 12, "constraint_one")
    prob += (-x + y <= 3, "constraint_two")
    prob += (x >= 4, "constraint_three")
    prob += (y <= 3, "constraint_four")
    # print(prob.constraints)

    # Add objective function
    prob += 3 * x + 5 * y
    # print(prob.objective)

    # Solve the problem
    print(prob)
    prob.solve()

    # Print the results
    print("Status:", plp.LpStatus[prob.status])
    print("x:", x.varValue)
    print("y:", y.varValue)
    print("Objective:", prob.objective.value())


def lp_problem2():
    # A company manufactures two products X and Y.
    # The production facilities restrict production to a total of 50 units per day.
    # Each day, 20 man-hours are available in the assembly shop, and 32 man-hours in the paint shop.
    # Each unit of X requires 30 minutes in the assembly shop and 24 minutes in the paint shop.
    # The corresponding times for product Y are 10 minutes and 48 minutes respectively.
    # The contribution to fixed overheads and profits for each unit of product X is $9,
    # and for a unit of product Y is $12.
    # Write the objective function and the constraints for this problem.

    # Create a LP problem
    prob = plp.LpProblem("LP Problem 2", plp.LpMaximize)

    # Create variables
    x = plp.LpVariable("X", lowBound=0, cat="Integer")
    y = plp.LpVariable("Y", lowBound=0, cat="Integer")

    # Add constraints
    prob += x + y <= 50
    prob += 30 * x + 10 * y <= 20 * 60
    prob += 24 * x + 48 * y <= 32 * 60

    # Add objective function
    prob += 9 * x + 12 * y

    # Solve the problem
    print(prob)
    prob.solve()

    # Print the results
    print("Status:", plp.LpStatus[prob.status])
    print("X:", x.varValue)
    print("Y:", y.varValue)
    print("Objective:", prob.objective.value())

    # Status: Optimal
    # x: 20.0
    # y: 30.0
    # Objective: 540.0


def lp_problem3():
    # To obtain maximum yield from a field, a farmer requires at least 300 units of nitrate,
    # 240 units of potash and 90 units of phosphorus.
    # He can buy two compound fertilisers A and B.
    # Each kilo of fertiliser A contains 10 units of nitrates, 5 units of potash, and 6 units of phosphorus;
    # the corresponding number of units for fertiliser B are 5, 10 and 1 respectively.
    # If the cost of fertiliser A is $1.50 per kilo and $2.00 per kilo for fertiliser B,
    # find the minimum cost quantities to purchase.
    # Write the objective function and the constraints for this problem.

    # Create a LP problem
    prob = plp.LpProblem("LP Problem 3")

    # Create variables
    a = plp.LpVariable("A", lowBound=0, cat="Integer")
    b = plp.LpVariable("B", lowBound=0, cat="Integer")

    # Add constraints
    prob += 10 * a + 5 * b >= 300
    prob += 5 * a + 10 * b >= 240
    prob += 6 * a + 1 * b >= 90

    # Add objective function
    prob += 1.5 * a + 2 * b

    # Solve the problem
    print(prob)
    prob.solve()

    # Print the results
    print("Status:", plp.LpStatus[prob.status])
    print("A:", a.varValue)
    print("B:", b.varValue)
    print("Objective:", prob.objective.value())

    # Status: Optimal
    # A: 24.0
    # B: 12.0
    # Objective: 60.0


# Linear Regression
def linear_r_problem1():
    plt.rcParams["figure.figsize"] = (5, 5)

    # Read data
    file_path = "data/Data_SLR.csv"
    df = pd.read_csv(file_path)
    print(df)
    print()

    # Plot correlation matrix
    correlation = df.corr()
    print(correlation)
    print()
    correlation_styler = correlation.style
    correlation_styler.format(precision=2)
    correlation_styler.background_gradient(cmap="coolwarm")
    correlation_styler.to_html("output/correlation.html")
    # The correlation between LSTAT and PRICE is -0.74.
    # This indicates that there is a negative correlation between LSTAT and PRICE.
    # When there is an increase in LSTAT, the PRICE decreases.

    # Plot scatter plot
    plt.scatter(x=df.LSTAT, y=df.PRICE)
    plt.title("LSTAT vs PRICE")
    plt.xlabel("$LSTAT$")
    plt.ylabel("$PRICE$")
    plt.show()

    # Split data into training and test sets
    x, y = df[["LSTAT"]], df.PRICE
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=5
    )
    print("Shape of x_train:", x_train.shape)
    print("Shape of x_test:", x_test.shape)
    print("Shape of y_train:", y_train.shape)
    print("Shape of y_test:", y_test.shape)
    print()

    # Create linear regression model
    model = LinearRegression()
    model.fit(x_train, y_train)
    print("Y-intercept (b0):", model.intercept_)
    print("Slope (b1):", model.coef_)
    print(f"Equation: y = {model.intercept_} {model.coef_[0]:+} * x")
    print()

    # R squared
    r2_score = model.score(x_test, y_test)
    print("R-squared:", r2_score)
    print()

    # Model parameters
    print(model.get_params())
    print()

    # Mean squared error
    y_test_pred = model.predict(X=x_test)
    mse = mean_squared_error(y_test, y_test_pred)
    print("Mean squared error:", mse)
    print()

    # Plot the fitted regression line
    plt.scatter(x_train, y_train, color="black")
    x_train_pred = model.predict(x_train)
    plt.plot(x_train, x_train_pred, color="blue", linewidth=3)
    plt.title("Linear Regression of LSTAT vs PRICE")
    plt.xlabel("$LSTAT$")
    plt.ylabel("$PRICE$")
    plt.show()


# Multiple Linear Regression
def linear_r_problem2():
    # Read data
    file_path = "data/boston.csv"
    df = pd.read_csv(file_path)
    print(df)
    print()

    # Plot correlation matrix
    correlation = df.corr()
    print(correlation)
    print()

    # Split data into training and test sets
    x, y = df.drop(columns="PRICE"), df.PRICE
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=5
    )
    print("Shape of x_train:", x_train.shape)
    print("Shape of x_test:", x_test.shape)
    print("Shape of y_train:", y_train.shape)
    print("Shape of y_test:", y_test.shape)
    print()

    # Normalisation
    scalar = StandardScaler()
    x_train_scaled = scalar.fit_transform(x_train)
    x_test_scaled = scalar.transform(x_test)

    # Create linear regression model
    model = LinearRegression()
    model.fit(x_train_scaled, y_train)
    print("Y-intercept (b0):", model.intercept_)
    print("Slope (b1):", model.coef_)
    print()

    # R squared
    r2_score = model.score(x_test_scaled, y_test)
    print("R-squared:", r2_score)
    print()

    # Mean squared error
    y_test_pred = model.predict(X=x_test_scaled)
    mse = mean_squared_error(y_test, y_test_pred)
    print("Mean squared error:", mse)
    print()


# SGD Regression
def sgd_linear_r_problem1():
    plt.rcParams["figure.figsize"] = (5, 5)

    # Read data
    file_path = "data/Data_SLR.csv"
    df = pd.read_csv(file_path)
    print(df)
    print()

    # Split data into training and test sets
    x, y = df[["LSTAT"]], df.PRICE
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=5
    )
    print("Shape of x_train:", x_train.shape)
    print("Shape of x_test:", x_test.shape)
    print("Shape of y_train:", y_train.shape)
    print("Shape of y_test:", y_test.shape)
    print()

    # Create Stochastic Gradient Descent model
    model = SGDRegressor(random_state=5)
    model.fit(x_train, y_train)
    print("Y-intercept (b0):", model.intercept_)
    print("Slope (b1):", model.coef_)
    print(f"Equation: y = {model.intercept_[0]} {model.coef_[0]:+} * x")
    print()

    # R squared
    r2_score = model.score(x_test, y_test)
    print("R-squared:", r2_score)
    print()

    # Mean squared error
    y_test_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_test_pred)
    print("Mean squared error:", mse)
    print()


def grid_search_sgd_linear_r_problem1():
    plt.rcParams["figure.figsize"] = (5, 5)

    # Read data
    file_path = "data/Data_SLR.csv"
    df = pd.read_csv(file_path)
    print(df)
    print()

    # Split data into training and test sets
    x, y = df[["LSTAT"]], df.PRICE
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=5
    )
    print("Shape of x_train:", x_train.shape)
    print("Shape of x_test:", x_test.shape)
    print("Shape of y_train:", y_train.shape)
    print("Shape of y_test:", y_test.shape)
    print()

    # Create Stochastic Gradient Descent model
    model = SGDRegressor()
    print("Parameters:", list(model.get_params()))
    print()

    # Create Grid Search
    # TODO: Add tol, random_state, learning_rate
    params = {
        "alpha": [1e-6, 1e-5, 0.0001, 0.001, 0.01, 0.1, 1],
        "learning_rate": ["constant", "invscaling", "optimal", "adaptive"],
        "loss": ["squared_error", "huber"],
        "max_iter": [5000, 10000, 20000],
        "penalty": ["l2", "l1", "elasticnet"],
        "random_state": [1, 2, 3, 4, 5],
        "tol": [1e-4, 1e-3, 1e-2],
    }
    clf = GridSearchCV(model, params)
    clf.fit(x_train, y_train)
    print("Best parameters:", clf.best_params_)
    best_estimator = clf.best_estimator_
    print("Best y-intercept (b0):", best_estimator.intercept_)
    print("Best slope (b1):", best_estimator.coef_)
    print("Best R-squared score:", clf.best_score_)
    print()
    # Best parameters: {'alpha': 0.001, 'learning_rate': 'optimal', 'loss': 'huber', 'max_iter': 5000, 'penalty': 'l2', 'random_state': 5, 'tol': 0.0001}
    # Best y-intercept (b0): [32.01114587]
    # Best slope (b1): [-1.19238353]
    # Best R-squared score: 0.5259240072725243


# Non-linear Regression
def non_linear_r_problem1():
    # Read data
    file_path = "data/NLData1.csv"
    df = pd.read_csv(file_path)
    print(df)
    print()

    # Correlation
    correlation = df.corr()
    print("Correlation:")
    print(correlation)
    print()

    x, y = df[["x"]], df.y

    # Scatter plot
    plt.scatter(x, y)
    plt.title("Scatter plot of x against y")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    # Polynomial Feature
    poly_feature = PolynomialFeatures(include_bias=False)
    x_poly = poly_feature.fit_transform(x)
    print("Polynomial features:")
    print(x_poly)
    print()

    # Create Linear Regression model
    model = LinearRegression()
    model.fit(x_poly, y)
    print("Y-intercept (b0):", model.intercept_)
    print("Slope (b1):", model.coef_)
    print(
        f"Equation: y = {model.intercept_} {model.coef_[0]:+} * x {model.coef_[1]:+} * x^2"
    )
    print()

    # R squared
    r2_score = model.score(x_poly, y)
    print("R-squared:", r2_score)
    print()

    # Mean squared error
    y_test_pred = model.predict(x_poly)
    mse = mean_squared_error(y, y_test_pred)
    print("Mean squared error:", mse)
    print()

    # Prediction
    x_pred = np.linspace(-5, 5, 100).reshape(100, 1)
    x_pred_poly = poly_feature.fit_transform(x_pred)
    y_pred = model.predict(x_pred_poly)
    plt.plot(x, y, "b.", label="Original data")
    plt.plot(x_pred, y_pred, "r-", label="Predicted data")
    plt.title("Prediction")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="upper left")
    plt.show()


def sgd_non_linear_r_problem2():
    # Read data
    file_path = "data/NLData1.csv"
    df = pd.read_csv(file_path)
    print(df)
    print()

    # Correlation
    correlation = df.corr()
    print("Correlation:")
    print(correlation)
    print()

    x, y = df[["x"]], df.y

    # Scatter plot
    plt.scatter(x, y)
    plt.title("Scatter plot of x against y")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    # Polynomial Feature
    poly_feature = PolynomialFeatures(include_bias=False)
    x_poly = poly_feature.fit_transform(x)
    print("Polynomial features:")
    print(x_poly)
    print()

    # Create Linear Regression model
    model = SGDRegressor()
    model.fit(x_poly, y)
    print("Y-intercept (b0):", model.intercept_)
    print("Slope (b1):", model.coef_)
    print(
        f"Equation: y = {model.intercept_[0]} {model.coef_[0]:+} * x {model.coef_[1]:+} * x^2"
    )
    print()

    # R squared
    r2_score = model.score(x_poly, y)
    print("R-squared:", r2_score)
    print()

    # Mean squared error
    y_test_pred = model.predict(x_poly)
    mse = mean_squared_error(y, y_test_pred)
    print("Mean squared error:", mse)
    print()

    # # Prediction
    # x_pred = np.linspace(-5, 5, 100).reshape(100, 1)
    # x_pred_poly = poly_feature.fit_transform(x_pred)
    # y_pred = model.predict(x_pred_poly)
    # plt.plot(x, y, 'b.', label="Original data")
    # plt.plot(x_pred, y_pred, 'r-', label="Predicted data")
    # plt.title("Prediction")
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.legend(loc="upper left")
    # plt.show()


# Logistic Regression
def logistic_r_problem1():
    file_path = "data/logdata.csv"
    df = pd.read_csv(file_path)
    print(df)
    print()

    x, y = df[["BMI"]], df.Outcome

    # Count each distinct values in y
    print(Counter(y))
    print()

    # Split data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=0
    )
    print("Shape of x_train:", x_train.shape)
    print("Shape of x_test:", x_test.shape)
    print("Shape of y_train:", y_train.shape)
    print("Shape of y_test:", y_test.shape)
    print()

    # Create Logistic Regression model
    model = LogisticRegression(
        random_state=0, solver="liblinear"
    )  # C, random_state, solver, max_iter
    model.fit(x_train, y_train)
    print("Classes:", model.classes_)
    print("Y-intercept (b0):", model.intercept_)
    print("Slope (b1):", model.coef_)
    print(f"Equation: ln(y) = {model.intercept_[0]} {model.coef_[0][0]:+} * x")
    print()

    # Accuracy
    accuracy = model.score(x_test, y_test)
    print("Accuracy:", accuracy)
    print()

    # Confusion matrix
    y_test_pred = model.predict(x_test)
    confusion_matrix_array = confusion_matrix(y_test, y_test_pred)
    print("Confusion matrix:")
    print(confusion_matrix_array)
    print()

    # Classification report
    print("Classification report:")
    print(classification_report(y_test, y_test_pred))
    print()


def logistic_r_problem2():
    file_path = "data/diabetes.csv"
    df = pd.read_csv(file_path)
    print(df)
    print()

    x, y = df.drop(columns="Outcome"), df.Outcome

    # Split data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=0
    )
    print("Shape of x_train:", x_train.shape)
    print("Shape of x_test:", x_test.shape)
    print("Shape of y_train:", y_train.shape)
    print("Shape of y_test:", y_test.shape)
    print()

    # Normalisation
    scalar = StandardScaler()
    x_train_scaled = scalar.fit_transform(x_train)
    x_test_scaled = scalar.transform(x_test)

    # Create Logistic Regression model
    model = LogisticRegression(
        random_state=0, solver="liblinear"
    )  # C, random_state, solver, max_iter
    model.fit(x_train_scaled, y_train)
    print("Classes:", model.classes_)
    print("Y-intercept (b0):", model.intercept_)
    print("Slope (b1):", model.coef_)
    print()

    # Accuracy
    accuracy = model.score(x_test_scaled, y_test)
    print("Accuracy:", accuracy)
    print()


# SGD Logistic Regression
def sgd_logistic_r_problem1():
    file_path = "data/logdata.csv"
    df = pd.read_csv(file_path)
    print(df)
    print()

    x, y = df[["BMI"]], df.Outcome

    # Split data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=0
    )
    print("Shape of x_train:", x_train.shape)
    print("Shape of x_test:", x_test.shape)
    print("Shape of y_train:", y_train.shape)
    print("Shape of y_test:", y_test.shape)
    print()

    # Create SGD Logistic Regression model
    model = SGDClassifier(loss="log", alpha=0.0001, max_iter=1000)
    model.fit(x_train, y_train)
    print("Y-intercept (b0):", model.intercept_)
    print("Slope (b1):", model.coef_)
    print()

    # Accuracy
    accuracy = model.score(x_test, y_test)
    print("Accuracy:", accuracy)


def grid_search_sgd_logistic_r_problem1():
    file_path = "data/logdata.csv"
    df = pd.read_csv(file_path)
    print(df)
    print()

    x, y = df[["BMI"]], df.Outcome

    # Split data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=0
    )
    print("Shape of x_train:", x_train.shape)
    print("Shape of x_test:", x_test.shape)
    print("Shape of y_train:", y_train.shape)
    print("Shape of y_test:", y_test.shape)
    print()

    # Create SGD Logistic Regression model
    model = SGDClassifier()
    print("Parameters:", list(model.get_params()))
    print()

    # Create grid search
    params = {
        "alpha": [1e-6, 1e-5, 0.0001, 0.001, 0.01, 0.1],
        "learning_rate": ["invscaling", "optimal", "adaptive"],
        "loss": ["hinge", "log_loss", "modified_huber", "squared_hinge", "perceptron"],
        "max_iter": [200, 500, 1000],
        "penalty": ["l2", "l1", "elasticnet"],
        "random_state": [1, 2, 3, 4, 5],
    }
    clf = GridSearchCV(model, params)
    clf.fit(x_train, y_train)
    print("Best parameters:", clf.best_params_)
    best_estimator = clf.best_estimator_
    print("Best y-intercept (b0):", best_estimator.intercept_)
    print("Best slope (b1):", best_estimator.coef_)
    print("Best R-squared score:", clf.best_score_)
    print()
    # Best parameters: {'alpha': 0.001, 'learning_rate': 'optimal', 'loss': 'squared_hinge', 'max_iter': 200, 'penalty': 'l2', 'random_state': 4}
    # Best y-intercept (b0): [-9.81805597e+13]
    # Best slope (b1): [[2.89458269e+12]]
    # Best R-squared score: 0.6719190404797601


if __name__ == "__main__":
    # optimisation_problem1()
    # optimisation_problem2()
    # lp_problem1()
    # lp_problem2()
    # lp_problem3()
    # linear_r_problem1()
    # linear_r_problem2()
    # sgd_linear_r_problem1()
    # grid_search_sgd_linear_r_problem1()
    # non_linear_r_problem1()
    # sgd_non_linear_r_problem2()
    # logistic_r_problem1()
    # logistic_r_problem2()
    # sgd_logistic_r_problem1()
    # grid_search_sgd_logistic_r_problem1()
    pass

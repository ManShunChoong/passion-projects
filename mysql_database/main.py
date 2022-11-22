import os
from typing import Optional, Union

import mysql.connector
import pandas as pd
import yaml
from mysql.connector.connection import MySQLConnection
from mysql.connector.cursor import MySQLCursorBuffered, MySQLCursorBufferedDict

dir_path = os.path.dirname(os.path.realpath(__file__))


class MySQLDatabase:
    """
    A class that manages database connection and execute queries.

    Attributes
    ----------
    con : MySQLConnection
        Connection to a MySQL Server.
    cursor : MySQLCursorBuffered
        A buffered cursor.
    dict_cursor : MySQLCursorBufferedDict
        A buffered cursor fetching rows as dictionaries.

    Notes
    -----
    Connector/Python Connection Establishment
    https://dev.mysql.com/doc/connector-python/en/connector-python-connectargs.html
    """

    def __init__(self):
        # Read configuration file
        file = "config.yaml"
        with open(f"{dir_path}/{file}", mode="r") as stream:
            try:
                config = yaml.safe_load(stream)
                user = config["mysql"]["user"]
                password = config["mysql"]["password"]
                database = config["mysql"]["database"]
                host = config["mysql"]["host"]

                print(f"Finish reading {file}")
            except yaml.parser.ParserError:
                raise yaml.parser.ParserError

        # Connect to MySQL server
        try:
            self.con = MySQLConnection(
                user=user, password=password, database=database, host=host
            )
            print("Connected to database")
        except mysql.connector.errors.ProgrammingError:
            raise mysql.connector.errors.ProgrammingError
        except mysql.connector.errors.DatabaseError:
            raise mysql.connector.errors.DatabaseError

        self.cursor = self.con.cursor(buffered=True)
        self.dict_cursor = self.con.cursor(buffered=True, dictionary=True)

    def execute_query(
        self,
        operation,
        params=(),
        multi=False,
        dictionary=False,
        commit=False,
        debug=False,
    ):
        """
        Execute the given operation substituting any markers with the given parameters, and
        return all rows of a query result set if there are.

        Parameters
        ----------
        operation : str
            Operation to be executed.
        params : tuple, default ()
            Parameters to substitute markers in operation.
        multi : bool, default False
            Execute multiple statements in one operation.
            If not set to True and multiple results are found, an InterfaceError will be raised.
        dictionary : bool, default False
            Fetch rows as dictionaries.
        commit : bool, default False
            Commit current transaction.
        debug : bool, default False
            Print operation and parameters.

        Returns
        -------
        Union[None, list[Optional[dict]], list[Optional[tuple]]]
            All rows of a query result set.

        See Also
        --------
        execute : Executes the given operation substituting any markers with the given parameters.

        Examples
        --------
        # >>> db = MySQLDatabase()
        # >>> db.execute_query("SELECT * FROM info WHERE column_name = %s", ("id",))
        # [('id', 'order', None), ('id', 'customer', None)]

        # >>> db = MySQLDatabase()
        # >>> db.execute_query("SELECT * FROM info WHERE column_name = %s", ("id",), dictionary=True)
        # [{'column_name': 'id', 'table_name': 'order', 'meta': None}, {'column_name': 'id', 'table_name': 'customer', 'meta': None}]
        """
        if debug:
            print(operation)
            if params:
                print(params)
        cursor = self.dict_cursor if dictionary else self.cursor
        cursor.execute(operation=operation, params=params, multi=multi)
        if commit:
            self.con.commit()
        if operation.startswith(("SELECT", "SHOW")):
            return cursor.fetchall()

    # Getter functions
    def get_table_names(self, debug=False):
        """
        Retrieve all table names in database.

        Parameters
        ----------
        debug : bool, default False
            Print operation.

        Returns
        -------
        list
            List of table names.
        """
        operation = "SHOW TABLES"
        result = self.execute_query(operation, debug=debug)
        return [row[0] for row in result]

    def get_column_names(self, table_name, data_type=False, debug=False):
        """
        Retrieve all column names in a table.

        Parameters
        ----------
        table_name : str
            Name of table.
        data_type : bool, default False
            Return column data types.
        debug : bool, default False
            Print operation and parameters.

        Returns
        -------
        Union[tuple[list, list], list]
            List of column names, with an optional list of data types.
        """
        if data_type:
            operation = (
                "SELECT COLUMN_NAME, DATA_TYPE "
                "FROM INFORMATION_SCHEMA.COLUMNS "
                "WHERE TABLE_NAME = %s"
            )
        else:
            operation = (
                "SELECT COLUMN_NAME "
                "FROM INFORMATION_SCHEMA.COLUMNS "
                "WHERE TABLE_NAME = %s"
            )
        params = (table_name,)
        result = self.execute_query(operation, params, debug=debug)
        if data_type:
            return [row[0] for row in result], [row[1] for row in result]
        else:
            return [row[0] for row in result]

    def get_table_data(self, table_name, dictionary=False, debug=False):
        """
        Retrieve all rows of data in a table.

        Parameters
        ----------
        table_name : str
            Name of table.
        dictionary : bool
            Fetch rows as dictionaries.
        debug : bool
            Print operation.

        Returns
        -------
        Union[list[dict], list[tuple]]
            List of row data.
        """
        operation = f"SELECT * FROM `{table_name}`"
        return self.execute_query(operation, dictionary=dictionary, debug=debug)

    # TODO: Revamp all methods below
    def print_table(self, table_name):
        if self.has_table(table_name):
            result = self.execute_query(
                f"SELECT * FROM `{table_name}`", dictionary=True
            )
            print(table_name)
            print("------------")
            print(pd.DataFrame(result))
        else:
            print(
                f"mysql.connector.errors.ProgrammingError: 1146 (42S02): Table '{table_name}' doesn't exist"
            )
        print()

    def print_database(self):
        query = "SHOW TABLES"
        result = self.execute_query(query)
        for row in result:
            self.print_table(table_name=row[0])

    # Boolean functions
    def has_table(self, table_name, debug=False):
        operation = "SHOW TABLES LIKE %s"
        data = (table_name,)
        result = self.execute_query(operation, data, debug=debug)
        return bool(result)

    def has_table_axis(self, table_name, operation, params, debug=False):
        if self.has_table(table_name):
            result = self.execute_query(operation, params, debug=debug)
            return bool(result)
        print(
            f"mysql.connector.errors.ProgrammingError: 1146 (42S02): Table '{table_name}' doesn't exist"
        )
        print()
        return False

    def has_column(self, table_name, column_name, debug=False):
        query = f"SHOW COLUMNS FROM `{table_name}` LIKE %s"
        data = (column_name,)
        return self.has_table_axis(
            table_name, operation=query, params=data, debug=debug
        )

    def has_row(self, table_name, value, column_name="id", debug=False):
        query = f"SELECT * FROM `{table_name}` WHERE {column_name} = %s"
        data = (value,)
        return self.has_table_axis(
            table_name, operation=query, params=data, debug=debug
        )

    # Table functions
    def add_table(self, table_name, debug=False):
        # Ref: https://www.techonthenet.com/mysql/primary_keys.php
        # Check if table_name exists
        if self.has_table(table_name):
            print(
                f"mysql.connector.errors.ProgrammingError: Table named '{table_name}' already exists"
            )
        else:
            query_1 = f"CREATE TABLE `{table_name}` (id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY)"
            query_2 = (
                "INSERT INTO info (column_name, table_name, meta) VALUES (%s, %s, %s)"
            )
            data_2 = ("id", table_name, None)
            self.execute_query(query_1, debug=debug)
            self.execute_query(query_2, data_2, commit=True, debug=debug)
            print(f"Table '{table_name}' successfully created.")
        print()

    def remove_table(self, table_name, debug=False):
        if self.has_table(table_name):
            query_1 = f"DROP TABLE `{table_name}`"
            query_2 = f"DELETE FROM info WHERE table_name = %s;"
            data_2 = (table_name,)
            self.execute_query(query_1, debug=debug)
            self.execute_query(query_2, data_2, commit=True, debug=debug)
            print(f"Table '{table_name}' successfully deleted.")
        else:
            print(
                f"mysql.connector.errors.ProgrammingError: 1146 (42S02): Table '{table_name}' doesn't exist"
            )
        print()

    # TODO: Column functions
    def add_column(
        self,
        table_name,
        column_name,
        data_type="varchar",
        referenced_table_name=None,
        debug=False,
    ):
        # Acceptable data types: varchar, int, double, timestamp, key
        data_type_dict = {
            "varchar": "VARCHAR(255)",
            "int": "INT",
            "double": "DOUBLE",
            "timestamp": "TIMESTAMP",
            "key": "INT UNSIGNED",
        }
        if not self.has_table(table_name):
            print(
                f"mysql.connector.errors.ProgrammingError: 1146 (42S02): Table '{table_name}' doesn't exist"
            )
        elif self.has_column(table_name, column_name):
            print(
                f"mysql.connector.errors.ProgrammingError: 1060 (42S21): Duplicate column name '{column_name}'"
            )
        elif (
            data_type == "key"
            and referenced_table_name
            and not self.has_table(referenced_table_name)
        ):
            print(
                f"mysql.connector.errors.DatabaseError: 1824 (HY000): Failed to open the referenced table '{table_name}'"
            )
        elif data_type not in data_type_dict:
            print(
                "mysql.connector.errors.ProgrammingError: 1064 (42000): You have an error in your SQL syntax; "
                f"check the manual that corresponds to your MySQL server version for the right syntax to use near '{column_name}'"
            )
        else:
            column_type = data_type_dict[data_type]
            query_1 = f"ALTER TABLE `{table_name}` ADD {column_name} {column_type}"
            query_2 = (
                "INSERT INTO info (column_name, table_name, meta) VALUES (%s, %s, %s)"
            )
            data_2 = (column_name, table_name, None)
            self.execute_query(query_1, commit=True, debug=debug)
            self.execute_query(query_2, data_2, commit=True, debug=debug)

            if data_type == "key" and referenced_table_name:
                query = f"ALTER TABLE `{table_name}` ADD FOREIGN KEY ({column_name}) REFERENCES {referenced_table_name}(id)"
                self.execute_query(query, commit=True, debug=debug)
            print(
                f"A column '{column_name}' {column_type} successfully created in '{table_name}'."
            )
        print()

    def modify_column(self):
        pass
        # ALTER TABLE table_name MODIFY COLUMN column_name datatype;

    def remove_column(self):
        pass
        # ALTER TABLE table_name DROP COLUMN column_name;

    # TODO: Row functions
    def add_row(self, table_name, column_names=("id",), values=None, debug=False):
        if not self.has_table(table_name):
            print(
                f"mysql.connector.errors.ProgrammingError: 1146 (42S02): Table '{table_name}' doesn't exist"
            )
        elif len(column_names) != len(values):
            print(
                f"mysql.connector.errors.DataError: 1136 (21S01): Column count doesn't match value count"
            )
        else:
            query = f"INSERT INTO `{table_name}` ({', '.join([col for col in column_names])}) VALUES (%s)"
            data = (values,)
            self.execute_query(query, data, commit=True, debug=debug)
            print(f"A row successfully created in '{table_name}'.")
        print()

    def remove_row(self, table_name, value, column_name="id", debug=False):
        if not self.has_table(table_name):
            print(
                f"mysql.connector.errors.ProgrammingError: 1146 (42S02): Table '{table_name}' doesn't exist"
            )
        elif not self.has_column(table_name, column_name):
            print(
                f"mysql.connector.errors.ProgrammingError: 1054 (42S22): Unknown column '{column_name}' in 'where clause'"
            )
        elif not self.has_row(table_name, value):
            print(f"Row ({column_name} = {value}) not found in '{table_name}'")
        else:
            query = f"DELETE FROM `{table_name}` WHERE {column_name} = %s"
            data = (value,)
            self.execute_query(query, data, commit=True, debug=debug)
            print(
                f"Rows ({column_name} = {value}) successfully deleted from '{table_name}'."
            )
        print()

    # TODO: Cell function
    def modify_cell(self, table_name, column_name, id_, value, debug=False):
        if not self.has_table(table_name):
            print(
                f"mysql.connector.errors.ProgrammingError: 1146 (42S02): Table '{table_name}' doesn't exist"
            )
        elif not self.has_column(table_name, column_name):
            print(
                f"mysql.connector.errors.ProgrammingError: 1054 (42S22): Unknown column '{column_name}' in 'where clause'"
            )
        elif not self.has_row(table_name, id_):
            print(f"Row (id = {id_}) doesn't exist in '{table_name}'")
        else:
            query = f"UPDATE `{table_name}` SET {column_name} = %s WHERE id = %s"
            data = (value, id_)
            self.execute_query(query, data, commit=True, debug=debug)
            print(
                f"Cell ({id_}, '{column_name}') successfully set to '{value}' in '{table_name}'."
            )
        print()

    # TODO: For debugging only
    def restart(self, add_info=True, debug=False):
        # restart everything from scratch

        # Get all table names
        query = "SHOW TABLES"
        result = self.execute_query(query)
        tables = [x[0] for x in result if x[0] != "info"]

        # Drop all tables
        self.execute_query("SET FOREIGN_KEY_CHECKS = 0")
        for table in tables:
            self.remove_table(table, debug=debug)
        self.execute_query("SET FOREIGN_KEY_CHECKS = 1")

        # Recreate table_info
        query = "DROP TABLE IF EXISTS `info`"
        self.execute_query(query, debug=debug)
        if add_info:
            query = """CREATE TABLE info 
			(column_name VARCHAR(50) NOT NULL,
			table_name VARCHAR(50) NOT NULL, 
			meta CHAR(100))
			"""
            self.execute_query(query, debug=debug)
            print("Table info successfully created!")

    def run_test(self):
        # self.restart(debug=True)
        self.print_table("info")

        table_name = "order"
        table_name_2 = "customer"

        self.add_table(table_name)
        self.add_column(table_name, column_name="name", data_type="varchar")
        self.modify_cell(table_name, column_name="name", id_=3, value="DOG")

        self.add_column(
            table_name,
            column_name="customer_id",
            data_type="key",
            referenced_table_name=table_name_2,
        )
        self.add_table(table_name_2)
        self.add_column(
            table_name,
            column_name="customer_id",
            data_type="key",
            referenced_table_name=table_name_2,
        )
        self.print_table("info")

        # self.add_row(table_name)
        # self.add_row(table_name)
        # self.add_row(table_name)
        # self.add_row(table_name, column_names=('id', 'name'), values=(None, 'ALI'))
        # self.add_row(table_name, column_names=('id', 'name'), values=(8, 'ALI'))
        # self.add_row(table_name, column_names=('name',), values=('ALI',))
        # self.add_row(table_name, column_names=('id', 'name'), values=('ALI',))
        self.modify_cell(table_name, column_name="name", id_=3, value="DOG")
        self.print_table(table_name)

        self.modify_cell(table_name, column_name="name", id_=1, value="DOG")
        self.modify_cell(table_name, column_name="name", id_=2, value="DOG")
        self.modify_cell(table_name, column_name="name", id_=3, value="DOG")

        self.remove_row(table_name, value=2)
        self.remove_row(table_name, value=9)
        self.remove_row(table_name, value=1, column_name="id")
        self.print_table(table_name)

        self.remove_table(table_name)
        self.remove_table(table_name_2)
        self.print_table("info")


if __name__ == "__main__":
    db = MySQLDatabase()
    # db.run_test()

# mysql.server start
# mysql -u bos -p
# SELECT @@datadir;

# mysqldump -u bos -p data > data.sql

# Note: SQL for selecting columns and their data types
# SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, COLUMN_TYPE
# FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = 'data';

# Note: SQL for selecting foreign keys and their referenced table column
# SELECT TABLE_NAME, COLUMN_NAME, CONSTRAINT_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
# FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
# WHERE REFERENCED_TABLE_SCHEMA = 'data';

# Note: Code for joining tables
# table1 = 'customer'
# table2 = 'order'
# primary_key = 'id'
# foreign_key = 'customer_id'
# f"SELECT * FROM {table1} JOIN {table2} ON {table1}.{primary_key} = {table2}.{foreign_key}"

# TODO: Change all print to logger.info()

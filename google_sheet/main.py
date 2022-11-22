import os
from string import ascii_uppercase
from typing import Optional, Union

import pandas as pd
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import Resource, build

dir_path = os.path.dirname(os.path.realpath(__file__))


def index_to_letter(index):
    """
    Convert numeric index to column letter in Google Sheet.

    Parameters
    ----------
    index : int
        Index of the column, starting from 0.

    Returns
    -------
    str
        Letter of the column position.

    Examples
    --------
    >>> index_to_letter(0)
    'A'

    >>> index_to_letter(25)
    'Z'

    >>> index_to_letter(26)
    'AA'

    >>> index_to_letter(60)
    'BI'
    """
    letters = []
    index += 1
    while index:
        index, digit = divmod(index - 1, 26)
        letters.insert(0, ascii_uppercase[digit])
    return "".join(letters)


class GoogleSheet:
    """
    A class that reads and writes Google Sheets.

    Parameters
    ----------
    spreadsheet_id : str
        Id of spreadsheet.
    sheet_name : str
        Name of sheet.
    column_row : int, default 1
        Index of the column row.
    last_cell : tuple[str, int]
        Indices of the most bottom right cell (default is ('A', 1), which implies no new values are read).

    Attributes
    ----------
    spreadsheets : Resource
        A class for interacting with Google Sheet resource.
    spreadsheet_id : str
    sheet_name : str
    column_row : int
    last_column : str
    last_row : int

    Notes
    -----
    Google Sheet documentation
    https://developers.google.com/sheets/api/reference/rest

    Video on accessing data to a spreadsheet using Google Sheets API
    https://youtu.be/4ssigWmExak
    """

    def __init__(self, spreadsheet_id, sheet_name, column_row=1, last_cell=("A", 1)):
        credentials = Credentials.from_service_account_file(
            filename=f"{dir_path}/token.json",
            scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"],
        )
        service = build(serviceName="sheets", version="v4", credentials=credentials)
        self.spreadsheets = service.spreadsheets()

        self.spreadsheet_id = spreadsheet_id
        self.sheet_name = sheet_name
        self.column_row = column_row
        self.last_column, self.last_row = last_cell

    def get_values(self, cell_range, dataframe=False):
        """
        Retrieve values from cell range given.

        Parameters
        ----------
        cell_range : str
            Cell range of values to be retrieved.
        dataframe : bool, default False
            Values are returned as a dataframe.

        Returns
        -------
        Union[pd.DataFrame, list, None]
            Values in the cell range.
        """
        request = self.spreadsheets.values().get(
            spreadsheetId=self.spreadsheet_id, range=f"{self.sheet_name}!{cell_range}"
        )
        response: dict = request.execute()
        if "values" in response:
            values: list = response["values"]
            return pd.DataFrame(values[1:], columns=values[0]) if dataframe else values
        return None

    def get_new_values(self, update=False, dataframe=False):
        """
        Retrieve new values from this sheet.

        Parameters
        ----------
        update : bool, default False
            New values will not be marked as new.
        dataframe : bool, default False
            Values are returned as a dataframe.

        Returns
        -------
        Union[pd.DataFrame, list, None]
            New values in this sheet.
        """
        # Get column values and last column index
        col_values = self.get_values(cell_range=f"{self.column_row}:{self.column_row}")
        last_column = index_to_letter(len(col_values[0]) - 1)
        if update:
            self.last_column = last_column

        # Get row values
        row_values = self.get_values(cell_range=f"A{self.last_row + 1}:{last_column}")
        if not row_values:
            return None

        if update:
            self.last_row += len(row_values)
        values = col_values + row_values
        if dataframe:
            df = pd.DataFrame(values, columns=col_values[0])
            return df.drop(index=0).reset_index(drop=True)
        return values

    def run_test(self):
        """
        Run testing on class methods.
        """
        print(self.get_values(cell_range="A:AM"))
        print(self.get_values(cell_range="A:AM", dataframe=True))

        # print(self.last_column, self.last_row)
        # print(self.get_new_values(dataframe=True))
        # print(self.last_column, self.last_row)
        # print(self.get_new_values(update=True, dataframe=True))
        # print(self.last_column, self.last_row)


if __name__ == "__main__":
    # cheras_sheet = GoogleSheet(
    #     spreadsheet_id="1H0lsUky4UOVGFqrHRXmqzhdZ-G0mOs7KKzLmSlwrBrs",
    #     sheet_name="Form Responses 1",
    #     last_cell=("AM", 24)
    # )
    TTDI_sheet = GoogleSheet(
        spreadsheet_id="1uVWmaCBxKJIh2WS7Z2M1_HOvn-AXAWsEemW0dQgU6qY",
        sheet_name="Form Responses 1",
        last_cell=("AM", 281),
    )
    # TTDI_sheet.run_test()

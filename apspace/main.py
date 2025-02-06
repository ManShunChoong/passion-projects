import re
from sys import argv
from typing import Dict

import requests
import yaml


def login_and_take_attendance(index: int, count: int, account: Dict, otp: str) -> None:
    try:
        username = account["username"]
        password = account["password"]
        api_key = account["api_key"]
        print(f"{index}/{count}")
        print(f"username: {username}")
    except KeyError as e:
        print(f"ERROR: Missing key in account: {account}")
        print(e)
        return

    try:
        # Get general ticket (login)
        url = "https://cas.apiit.edu.my/cas/v1/tickets"
        data = {"username": username, "password": password}
        response = requests.post(url, data)

        pattern = r'TGT-[^"]*'
        match = re.search(pattern, response.text)
        general_ticket = match.group()

        print(f"general_ticket: {general_ticket}")
    except Exception as e:
        print(f"ERROR: Failed to get general ticket for {username}")
        print(e)
        return

    try:
        # Get attendance service ticket
        url = f"https://cas.apiit.edu.my/cas/v1/tickets/{general_ticket}?service=https://api.apiit.edu.my/attendix"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        response = requests.post(url, headers=headers)
        service_ticket = response.text

        print(f"service_ticket: {print(service_ticket)}")
    except Exception as e:
        print(f"ERROR: Failed to get service ticket for {username}")
        print(e)
        return

    try:
        # Take attendance
        url = "https://attendix.apu.edu.my/graphql"
        json = {
            "operationName": "updateAttendance",
            "query": "mutation updateAttendance($otp: String!) {\n  updateAttendance(otp: $otp) {\n    id\n    attendance\n    classcode\n    date\n    startTime\n    endTime\n    classType\n    __typename\n  }\n}\n",
            "variables": {"otp": otp},
        }
        headers = {"Ticket": service_ticket, "x-api-key": api_key}
        response = requests.post(url, json=json, headers=headers)

        response_dict = response.json()
        if "errors" in response_dict:
            messages = (error["message"] for error in response_dict["errors"])
            print(f"attendance: {'; '.join(messages)}")
        else:
            print(f"attendance: {response_dict}")
    except Exception as e:
        print(f"ERROR: Failed to take attendance for {username}")
        print(e)


def main() -> None:
    otp = argv[1]
    print(f"otp: {otp}")
    print()

    file_path = "accounts.yaml"
    with open(file_path) as file:
        accounts = yaml.safe_load(file)
    count = len(accounts)

    for index, account in enumerate(accounts, start=1):
        login_and_take_attendance(index, count, account, otp)
        print()
    print("Done")


if __name__ == "__main__":
    main()

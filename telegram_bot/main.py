import os
from pathlib import Path
from typing import IO, Optional, Union

import yaml
from telegram import Bot, InputFile, Message, ParseMode, PhotoSize, Update

dir_path = os.path.dirname(os.path.realpath(__file__))


class TelegramBot:
    def __init__(self):
        # Read configuration file
        file = "config.yaml"
        with open(f"{dir_path}/{file}", mode="r") as stream:
            try:
                config = yaml.safe_load(stream)
                telegram_token = config["telegram"]["token"]

                print(f"Finish reading {file}")
            except yaml.parser.ParserError as e:
                print(f"yaml.parser.ParserError: {e}")
                exit(-1)

        self.telegram_bot = Bot(token=telegram_token)
        print("Connected to Telegram bot")

    def get_updates(self) -> list[Update]:
        return self.telegram_bot.get_updates()

    def send_message(self, chat_id: Union[int, str], text: str) -> Message:
        return self.telegram_bot.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode=ParseMode.MARKDOWN,
        )

    def send_image(
        self,
        chat_id: Union[int, str],
        image: Union[str, bytes, IO, InputFile, Path, PhotoSize],
        caption: Optional[str] = None,
    ) -> Message:
        return self.telegram_bot.send_photo(
            chat_id=chat_id,
            photo=image,
            caption=caption,
            parse_mode=ParseMode.MARKDOWN,
        )

    def run_test(self):
        print(self.send_message(chat_id="1049219239", text="text2"))

        updates = self.get_updates()
        for update in updates:
            print(update.effective_message)


if __name__ == "__main__":
    telegram_bot = TelegramBot()
    telegram_bot.run_test()

# https://api.telegram.org/bot1833210921:AAFK-VDXOV5irTvHyun-r8LOMrtUUrSJZXQ/getUpdates

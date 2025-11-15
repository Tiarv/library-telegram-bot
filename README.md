# library-telegram-bot

Another telegram bot that provides interface to INPX-based libraries

(libraries themselves are not included, procure them on your own)

## Installation:

```
# adduser library
# su - library
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install --upgrade pip
$ pip install python-telegram-bot
$ git clone https://github.com/Tiarv/library-telegram-bot.git
$ cd library-telegram-bot.git
# populate bot.conf with your credentials
$ vim bot.conf
$ python3 ./bot.py
```

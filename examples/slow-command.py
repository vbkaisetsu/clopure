#!/usr/bin/env python3

import sys
import time
import os

"""
[ サーバー側の時間のかかるプログラム ]

* 初期化に時間がかかる（処理のたびにプロセスを立ち上げられない）
* 一対一入出力

------
標準入力で指定された秒数待機し，
プロセスIDと待機秒数を標準出力に書き出す
------
"""

print("pid %d, initialized" % os.getpid(), file=sys.stderr)

for line in sys.stdin:
    try:
        sec = int(line)
    except:
        sec = 1
    time.sleep(sec)
    print("pid %d, wait %d seconds" % (os.getpid(), sec))
    sys.stdout.flush()

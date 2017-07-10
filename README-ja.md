# clopure
Pythonで実装されたマルチプロセッシング対応関数型言語

## インストール

```
$ sudo pip3 install clopure
```
または
```
$ git clone https://github.com/vbkaisetsu/clopure.git
$ cd clopure
$ ./setup.py test
$ ./setup.py build
$ sudo ./setup.py install
```

## clopure を対話型コンソールで起動する

引数無しで clopure を起動します:
```
$ clopure
```

`>>>` に続けてコードを入力します。<kbd>Enter</kbd> または <kbd>Return</kbd> キーを押すとコードを実行できます。
```clojure
>>> ((fn fact [n] (if (> n 0) (* n (fact (- n 1))) 1)) 5)
120
```

入力されたコードのカッコが閉じていない場合は，追加のコードの入力を求められます。
```clojure
>>> (list (map #(+ % 1)
...            [1 2 3 4 5]))
[2, 3, 4, 5, 6]
```

## clopure スクリプトファイルを実行する

スクリプトファイルを指定することもできます
```
$ clopure script.clp
```
または
```
$ echo "(print (+ 1 1))" | clopure
```

## マルチプロセッシング

いくつかの関数はマルチプロセッシングに対応しています。これを有効化するには，clopure に `-p` オプションを付けて実行します。
例えば，以下のコマンドは clopure を最大4プロセスで実行します。
```
$ clopure -p 4
```

現在，clopure には4つのマルチプロセッシング関数があります: `pmap`, `pmap-unord` `iter-mp-split`, `iter-mp-split-unord`.
`pmap` は `map` 関数のマルチプロセッシング対応版です。

`iter-mp-split` は map 風の関数を引数にとって，新たな map 風の関数を返します。
(「map 風の関数」とは，イテレーターをとってイテレーターを返す関数のことです。)

`iter-mp-split` が呼ばれると，指定されたプロセス数だけ与えられた関数を実行し，
また `iter-mp-split` によって生成された関数に与えられたイテレーターのアイテムは，各プロセスに振り分けられます。

以下のコードは `iter-mp-split` の例です。
```clojure
>>> (defimport random random)
>>> (defimport time sleep)
>>> (defimport os getpid)
>>> (list ((iter-mp-split ; Creates a new map like function
...          #(do (sleep (random))                      ; Pseudo initialization
...               (print "pid:" (getpid) "initialized") ;
...               (map #(do (sleep (random))            ; Pseudo one-by-one processing
...                         (print "pid:" (getpid) %)   ;
...                         (+ % 1))                    ;
...                    %)))                             ;
...        (range 10))) ; Generates numbers from 0 to 9
```
出力は
```
pid: 51694 initialized
pid: 51693 initialized
pid: 51694 0
pid: 51695 initialized
pid: 51694 2
pid: 51694 4
pid: 51696 initialized
pid: 51694 5
pid: 51693 1
pid: 51693 8
pid: 51693 9
pid: 51695 3
pid: 51696 6
pid: 51694 7
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

# clopure
Multi-processing supported functional language for Python

## Install

```
$ git clone https://github.com/vbkaisetsu/clopure.git
$ cd clopure
$ ./setup.py build
$ sudo ./setup.py install
```

## Launch clopure with the interactive console

Just run clopure without arguments:
```
$ clopure
```

You can input codes after `>>>`. If you press <kbd>Enter</kbd> or <kbd>Return</kbd>, you can run a code.
```clojure
>>> ((fn fact [n] (if (> n 0) (* n (fact (- n 1))) 1)) 5)
120
```

When brackets of the input code is not closed, the console requires additional lines.
```clojure
>>> (list (map #(+ % 1)
...            [1 2 3 4 5]))
[2, 3, 4, 5, 6]
```

## Run a clopure script file

You can also specify a script file:
```
$ clopure script.clp
```
or
```
$ echo "(print (+ 1 1))" | clopure
```

## Multi-processing

Some functions support multi-processing. If you want to enable it, prease run clopure with `-t` option.
For example, the following command launches clopure with 4 processes:
```
$ clopure -t 4
```

Currently, clopure has 4 multi-processing functions: `pmap`, `pmap-unord` `iter-mp-split` and `iter-mp-split-unord`.
`pmap` is the multi-processing version of `map` function.

`iter-mp-split` takes a map like function and returns a new map like function
('map like function' means that it takes an iterator and also returns an iterator.)

When `iter-mp-split` is called, it starts a given function on the specified number of processes,
and items of the given iterator that are passed to the function created by `iter-mp-split` are distributed to each process.

The following code is an example of `iter-mp-split`:
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
The output is:
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

When you use `iter-mp-split`, the initialization process of the given function is run only for each process.

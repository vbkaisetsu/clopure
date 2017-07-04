(defimport builtins list)
(defimport builtins print)
(defimport time sleep)

(doseq
  [x (pmap
    (fn [x]
      (do
        (sleep 0.5)
        (print "+ x 1:" x)
        (+ x 1)))
    (pmap
      (fn [x]
        (do
          (print "* x x:" x)
          (* x x)))
      [0 1 2 3 4 5 6 7 8 9]))]
  (print x))

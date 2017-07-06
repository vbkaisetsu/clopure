(defimport time sleep)

(doseq
  [x (pmap
    #(do
      (sleep 0.5)
      (print "+ x 1:" %)
      (+ % 1))
    (pmap
      #(do
        (print "* x x:" %)
        (* % %))
      [0 1 2 3 4 5 6 7 8 9]))]
  (print x))

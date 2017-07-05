(defimport-as infcount itertools count)

(list
  (map #(print ((. % decode) "utf-8"))
        ((iter-split
            #(system-map "python3 test.py" %))
         (map (fn [_] ((. (+ (input) "\n") encode) "utf-8")) (infcount 0)))))

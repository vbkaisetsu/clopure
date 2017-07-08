(defimport-as infcount itertools count)

(def command (input "Command: "))
(print "Please input some lines, and press Ctrl+D")

(print
  (list
    (map #((. % decode) "utf-8")
          ((iter-mp-split
             #(system-map command %))
           (map #((. % encode) "utf-8") (take-while bool (map (fn [_] (read-line)) (infcount 0))))))))

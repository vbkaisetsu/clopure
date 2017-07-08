(defimport listen_iter listen_iter_connection)

(def command (input "Command: "))

(dorun
  (map #(extract-args #(do ((. %2 send) %1)
                           (print "sent:" ((. %1 decode) "utf-8")))
                      %)
        (extract-args (iter-mp-split-unord
                        #(zip (system-map command %1) %2))
                      (unzip
                        (map #(arg-list ((. % recv) 1024) %)
                              (listen_iter_connection "localhost" 1234 3))
                        2))))

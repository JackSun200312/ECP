(set-logic QF_NIA)

; Declare constant for instance
(declare-const n Int)

; Helper functions
(define-fun factorial ((x Int)) Int
    (ite (<= x 1) 1 (* x (factorial (- x 1)))))

(define-fun is_prime ((p Int)) Bool
    (and (>= p 2)
         (forall ((d Int))
             (or (= d 1) (= d p) (not (= (mod p d) 0))))))

(define-fun product_sum ((n Int)) Int
    (let ((primes (filter is_prime (range 2 (+ n 1)))))
        (reduce * (map + (combinations primes 2)))))

; Constraints
(assert (> n 2))
(assert (= (mod (product_sum n) (factorial n)) 0))

; Check satisfiability
(check-sat)
(get-value (n))

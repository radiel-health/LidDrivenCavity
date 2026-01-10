#lang racket
(require math/statistics)

(define (square x) (* x x))

;; Euclidean distance (x and y are lists that represent vectors (in the linear algebra sense))
(define (dist x y) (sqrt (apply + (map square (map - x y)))))

;; Calculates the average euclidean distance between a point (list) and a list of points
(define (average-dist points x)
  (/ (apply + (map (λ (y) (dist x y)) points)) (length points)))

(define pts '((1 1) (1 2) (2 1)))

(define new-pts '((1.1 1) (1.2 1) (1.3 1) (1.4 1) (1.5 1)
       (1.6 1) (1.7 1) (1.8 1) (1.9 1)))

(define res (map (λ (x) (cons x (average-dist pts x))) new-pts))

;; Results are sorted by the average-dist
(define sorted-res (sort res #:key cdr <))

(define sorted-res-avg-dists (map cdr sorted-res))

(define rmse-re-100 '(3.36e-07 3.37e-07 3.39e-07 3.39e-07 3.41e-07
  3.42e-07 3.43e-07 3.44e-07 3.46e-07))

(correlation sorted-res-avg-dists rmse-re-100)

(define mae-re-100 '(3.33e-07 3.33e-07 3.35e-07 3.36e-07 3.37e-07 3.38e-07 3.39e-07 3.40e-07 3.41e-07))

(correlation sorted-res-avg-dists mae-re-100)

(define rmse-re-1000 '(2.68e-07 2.70e-07 2.72e-07 2.74e-07 2.77e-07 2.79e-07 2.82e-07 2.85e-07 2.88e-07))

(correlation sorted-res-avg-dists rmse-re-1000)

(define mae-re-1000 '(2.58e-07 2.60e-07 2.62e-07 2.66e-07 2.70e-07 2.73e-07 2.78e-07 2.81e-07 2.85e-07))

(correlation sorted-res-avg-dists mae-re-1000)

(define rmse-re-2100 '(2.68e-07 2.64e-07 2.63e-07 2.65e-07 2.68e-07 2.72e-07 2.78e-07 2.84e-07 2.92e-07))
(correlation sorted-res-avg-dists rmse-re-2100)

(define mae-re-2100 '(2.53e-07 2.50e-07 2.50e-07 2.51e-07 2.53e-07 2.57e-07 2.67e-07 2.73e-07 2.83e-07))
(correlation sorted-res-avg-dists mae-re-2100)

#|
Output:
0.9793591384640998
0.9746694060195372
0.9938205010467132
0.9926655148786071
0.9521327320401638
0.9500599960187426
|#
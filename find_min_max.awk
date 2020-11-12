BEGIN {
    first = 0
}

!/^#/ && NF > 0 {
    if (first == 0) {
        max = $NF
        min = $NF
        first = 1
    }
    if (first > 0) {
        if (max < $NF) {
            max = $NF
        }
        if (min > $NF) {
            min = $NF
        }
    }
}

END {
    printf("%15.7f\n", max)
    printf("%15.7f\n", min)
}

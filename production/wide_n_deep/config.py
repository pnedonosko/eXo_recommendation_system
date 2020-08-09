# one minute = 60 secs
_ONE_MIN = 60
_ONE_HOUR = 60 * _ONE_MIN

# define random values intervals
_PEAK_INTERVAL_LONG = (10 * _ONE_MIN, 4 * _ONE_HOUR)
_NONPEAK_INTERVAL_LONG = (30 * _ONE_MIN, 8 * _ONE_HOUR)

_DNN_HIDDEN_UNITS = [256, 512, 256, 256, 128, 64, 32, 16]


# define start of peak time for each day
_day2peak = {
    "Monday" : 8,
    "Tuesday" : 13,
    "Wednesday" : 14,
    "Thursday" : 9,
    "Friday" : 14
}


_weekday2number = {
    "Monday" : 1,
    "Tuesday" : 2,
    "Wednesday" : 3,
    "Thursday" : 4,
    "Friday" : 5,
    "Saturday" : 6,
    "Sunday" : 7
}


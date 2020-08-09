from config import _ONE_MIN, _ONE_HOUR, _PEAK_INTERVAL_LONG, _NONPEAK_INTERVAL_LONG, _day2peak, _weekday2number

from tensorflow import feature_column

import datetime
import random
import pandas as pd



def get_date(secs):
  return datetime.datetime.fromtimestamp(secs).strftime("%A, %d/%m/%Y %H:%M:%S")

def get_date_separated(secs):
    date = datetime.datetime.fromtimestamp(secs)
    # return year, month, weekday, date(day),  hour, min, secs
    return int(date.strftime("%Y")), int(date.strftime("%m")), date.strftime("%A"), int(date.strftime("%d")), int(date.strftime("%H")), int(date.strftime("%M")), int(date.strftime("%S"))


def is_peak(secs):
    date = get_date_separated(secs)
    weekday = date[2]

    # take start of the peak time from the date in seconds
    start = _day2peak[weekday] * _ONE_HOUR
    # define end as 3 hours after start
    end = start + 3 * _ONE_HOUR

    # define current time in terms of seconds - seconds in hours + seconds in minutes and seconds
    current_secs = (_ONE_HOUR * date[4]) + (_ONE_MIN * date[5]) + date[6]

    return start <= current_secs <= end


def in_working_time(secs):
    date = get_date_separated(secs)

    # define start and end of the working day, i.e 8 AM and 18 PM
    business_time_start = 8 * _ONE_HOUR
    business_time_end = 18 * _ONE_HOUR

    # define current time in terms of seconds - seconds in hours + seconds in minutes and seconds
    current_secs = (_ONE_HOUR * date[4]) + (_ONE_MIN * date[5]) + date[6]

    return business_time_start < current_secs < business_time_end


def is_weekend(secs):
    date = get_date_separated(secs)

    weekday = date[2]
    if weekday == 'Saturday' or weekday == 'Sunday':
        return True




def create_effective_age(posted_time, peak, nonpeak):
    effective_ages = []

    for t in posted_time:

        if is_weekend(t):
            effective_age = random.randint(nonpeak[0], nonpeak[1]) * 2
            effective_ages.append(effective_age)
            continue

        if is_peak(t):
            effective_age = random.randint(peak[0], peak[1])
        else:
            effective_age = random.randint(nonpeak[0], nonpeak[1])

        if not in_working_time(t + effective_age):
            effective_age *= 2

        effective_ages.append(effective_age)

    return pd.Series(effective_ages)

def create_weekday(posted_time):

    weekdays = []

    for t in posted_time:
        weekdays.append(_weekday2number[get_date_separated(t)[2]])

    return weekdays

def create_daytime(posted_time):

    daytimes = []

    for t in posted_time:
        daytimes.append(get_date_separated(t)[4])

    return daytimes


# functions to reduce code duplication
def participant_action(part_action):
    participant_action = feature_column.categorical_column_with_vocabulary_list(
        part_action, ['commented', 'liked', 'viewed'])
    return participant_action

def participant_focus(part_f):
    participant_focus = feature_column.categorical_column_with_vocabulary_list(
        part_f, ['engineering', 'sales', 'marketing', 'management', 'financial', 'other', 'none'])
    return participant_focus
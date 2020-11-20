#!/usr/bin/python
from string import Formatter
from timeit import default_timer as timer
from datetime import timedelta


# function to calculate, format and return string with human friendly elapsed time
def get_time_tick(start_timer):
    end_timer = timer()
    return str_from_delta(timedelta(seconds=end_timer - start_timer))


# format function to return human friendly timedelta value
def str_from_delta(t_delta, fmt='{H:02}h {M:02}m {S:02}s', input_type='timedelta'):
    """Convert a datetime.timedelta object or a regular number to a custom-
    formatted string, just like the str_from_delta() method does for datetime.datetime
    objects.

    The fmt argument allows custom formatting to be specified.  Fields can
    include seconds, minutes, hours, days, and weeks.  Each field is optional.

    Some examples:
        '{D:02}d {H:02}h {M:02}m {S:02}s' --> '05d 08h 04m 02s' (default)
        '{W}w {D}d {H}:{M:02}:{S:02}'     --> '4w 5d 8:04:02'
        '{D:2}d {H:2}:{M:02}:{S:02}'      --> ' 5d  8:04:02'
        '{H}h {S}s'                       --> '72h 800s'

    The input_type argument allows t_delta to be a regular number instead of the
    default, which is a datetime.timedelta object.  Valid input_type strings:
        's', 'seconds',
        'm', 'minutes',
        'h', 'hours',
        'd', 'days',
        'w', 'weeks'
    """
    
    # Convert t_delta to integer seconds.
    if input_type == 'timedelta':
        remainder = int(t_delta.total_seconds())
    elif input_type in ['s', 'seconds']:
        remainder = int(t_delta)
    elif input_type in ['m', 'minutes']:
        remainder = int(t_delta) * 60
    elif input_type in ['h', 'hours']:
        remainder = int(t_delta) * 3600
    elif input_type in ['d', 'days']:
        remainder = int(t_delta) * 86400
    elif input_type in ['w', 'weeks']:
        remainder = int(t_delta) * 604800
    
    f = Formatter()
    desired_fields = [field_tuple[1] for field_tuple in f.parse(fmt)]
    possible_fields = ('W', 'D', 'H', 'M', 'S')
    constants = {'W': 604800, 'D': 86400, 'H': 3600, 'M': 60, 'S': 1}
    values = {}
    for field in possible_fields:
        if field in desired_fields and field in constants:
            values[field], remainder = divmod(remainder, constants[field])
    return f.format(fmt, **values) + ' %sms' % str(t_delta.microseconds)[:-3]



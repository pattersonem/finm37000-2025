from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

def us_business_day():
    return CustomBusinessDay(calendar=USFederalHolidayCalendar())

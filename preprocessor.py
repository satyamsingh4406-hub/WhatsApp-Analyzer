import re
import pandas as pd
def preprocess(data):
    pattern = r'\d{1,2}/\d{1,2}/\d{2},\s\d{1,2}:\d{2}\s-\s'
    # pattern ke hisab se kiya apne chat ke data ke hisab se

    messages = re.split(pattern, data)[1:]
    # khali message alag kar liya split se
    dates = re.findall(pattern, data)
    # dates nikal li findall se

    df = pd.DataFrame({
        'user_message': messages,
        'message_date': dates
    })

    df['message_date'] = pd.to_datetime(
        df['message_date'],
        format='%m/%d/%y, %H:%M - '
        # python ke date format me change kardiya
    )

    df.rename(columns={'message_date': 'date'}, inplace=True)

    users = []
    messages = []

    for message in df['user_message']:
        entry = re.split('([\w\W]+?):\s', message)  # naam alag kar liya message alag
        if entry[1:]:  # user name
            users.append(entry[1])
            messages.append(entry[2])
        else:
            users.append('group_notification')
            messages.append(entry[0])

    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)

    df['only_date']=df['date'].dt.date
    df['year'] = df['date'].dt.year  # years alag se add kar liya
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()  # Month alag se add kar liya
    df['day'] = df['date'].dt.day  # ab day kar liya
    df['day_name']=df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour  # hour kar liya(24 hour wala)
    df['minute'] = df['date'].dt.minute  # ab minute

    period=[]
    for hour in df[['day_name', 'hour']]['hour']:
        if hour==23:
            period.append(str(hour)+"-"+str('00'))
        elif hour==0:
            period.append(str('00')+"-"+str(hour+1))
        else:
            period.append(str(hour)+"-"+str(hour+1))

    df['period'] = period

    return df

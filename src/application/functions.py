def get_rpe(speed):
    rpes = {
        10: 0.3,
        9: 0.5,
        8: 0.75,
        7: 1,
        6: 1.3
    }
    if speed > rpes[6]:
        return 5
    elif speed > rpes[7]:
        return 6
    elif speed > rpes[8]:
        return 7
    elif speed > rpes[9]:
        return 8
    elif speed > rpes[10]:
        return 9
    else:
        return 10


def generate_chart(speed, rpe):
    # print(get_rpe(speed))
    if get_rpe(speed) != rpe:
        rpes = {
            10: 0.3,
            9: 0.5,
            8: 0.75,
            7: 1,
            6: 1.3,
            5: 1.5,
            4: 1.75,
            3: 2,
            2: 2.25,
            1: 2.5
        }
        rpes[rpe] = round(speed, 2)
        i = rpe
        while i > 1:
            if rpes[i] >= rpes[i - 1]:
                rpes[i - 1] = rpes[i - 1] + 0.2
            i -= 1
        rpe_chart = []
        rpe_items = rpes.items()
        for item in rpe_items:
            rpe_chart.append(item[1])
        print(rpe_chart)
        rpe_chart.reverse()
        return rpe_chart
    else:
        return [2.5, 2.25, 2, 1.75, 1.5, 1.3, 1, 0.75, 0.5, 0.3]


def string_to_float_list(string):
    string = string.split(',')
    for i in range(0, len(string)):
        string[i] = float(string[i].strip())
    return string


def float_list_to_string(float_list):
    return(','.join(str(x) for x in float_list))

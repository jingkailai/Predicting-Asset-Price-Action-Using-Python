listarima = [("DBS",0.1,0.05),("STI",0.2,0.06),("Singtel",0.3,0.07),("UOB",0.3,0.01),("OCBC",0.9,0.1),("Starhub",0.8,0.99)]
listlstm = [("DBS",0.2,0.01),("STI",0.3,0.005),("Singtel",0.1,0.2),("UOB",0.8,0.02),("OCBC",0.95,0.15),("Starhub",0.7,0.99)]


def compare_and_sort(listmodel):

    listmodel.sort(reverse=True, key=lambda list: list[2])
    return listmodel[:5]



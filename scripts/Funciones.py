
def recode_category(category):
    if category in ['0s=suspect Blood Donor', '0=Blood Donor']:
        return 0
    elif category in ['1=Hepatitis', '2=Fibrosis', '3=Cirrhosis']:
        return 1
    else:
        return category  # o algún valor predeterminado como -1 o np.nan para categorías que no coinciden



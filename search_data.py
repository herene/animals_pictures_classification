import requests
import os
def getManyPages(keyword,pages):
    params=[]
    for i in range(0,30*pages+30,30):
        params.append({
                      'tn': 'resultjson_com',
                      'ipn': 'rj',
                      'ct': 201326592,
                      'is': '',
                      'fp': 'result',
                      'queryWord': keyword,
                      'cl': 2,
                      'lm': -1,
                      'ie': 'utf-8',
                      'oe': 'utf-8',
                      'adpicid': '',
                      'st': -1,
                      'z': '',
                      'ic': 0,
                      'word': keyword,
                      's': '',
                      'se': '',
                      'tab': '',
                      'width': '',
                      'height': '',
                      'face': 0,
                      'istype': 2,
                      'qc': '',
                      'nc': 1,
                      'fr': '',
                      'pn': i,
                      'rn': 30,
                      'gsm': '3c',
                      '1488942260214': ''
                  })
    url = 'https://image.baidu.com/search/acjson'
    urls = []
    for i in params:
        urls.append(requests.get(url,params=i).json().get('data'))
    return urls

def getImg(dataList, localPath):
    if not os.path.exists(localPath):  # 新建文件夹

        os.mkdir(localPath)

    x = 0

    for list in dataList:

        for i in list:

            if i.get('thumbURL') != None:

                print('正在下载：%s' % i.get('thumbURL'))

                ir = requests.get(i.get('thumbURL'))

                open(localPath + '%d.jpg' % x, 'wb').write(ir.content)

                x += 1

            else:

                print('图片链接不存在')

if __name__ == '__main__':

    dataList = getManyPages('马',10)  
    getImg(dataList,'C:\\Users\\63558\\Downloads\\dataset\\horse') 

    dataList = getManyPages('猫',10)  
    getImg(dataList,'C:\\Users\\63558\\Downloads\\dataset\\cat') 
    
    dataList = getManyPages('狗',10)  
    getImg(dataList,'C:\\Users\\63558\\Downloads\\dataset\\dog') 
    
    dataList = getManyPages('大象',10)  
    getImg(dataList,'C:\\Users\\63558\\Downloads\\dataset\\elephant') 
    
    dataList = getManyPages('猴子',10)  
    getImg(dataList,'C:\\Users\\63558\\Downloads\\dataset\\monkey') 



    

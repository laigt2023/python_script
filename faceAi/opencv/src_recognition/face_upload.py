import os
import requests

# 已存在的人脸特征库列表
FACE_FEATURELIBS_MAP={}

def upload_files(directory_path, api_endpoint):
    global TOKEN

  


    # 遍历文件夹中的文件
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                one_info = get_featurelibs_data(filename)
        
                if one_info:
                    url = api_endpoint +"/" + one_info["id"] + "/import"

                    #dir_api_endpoint = api_endpoint +"/" + one_info["id"]
                    #dir_response = requests.get(dir_api_endpoint,{'Authentication': TOKEN})
                    
                    # 处理响应
                    #if dir_response.status_code == 200:
                       # print(f"{dir_api_endpoint} 请求成功！")

                        
                    # 打开文件并上传到指定接口
                    with open(file_path, 'rb') as file:
                        files = {'files': file}
                        response = requests.post(url, files=files,headers = {'Authentication': TOKEN})
                        # 处理响应
                        if response.status_code == 200:
                            print(f"文件 {filename} 上传成功！")
                            print(f"{url} : {response.text}")
                        else:
                            print(f"文件 {filename} 上传失败，错误码：{response.status_code}")
                        
                    #else:
                        #print(f"{dir_api_endpoint} 错误码：{response.status_code}")
                        #print(
                        # f"{dir_api_endpoint} : {response.text}")
                        #return



# 获取人脸特征库信息
def get_featurelibs_data(name):
    global FACE_FEATURELIBS_MAP

    # 获取到对应特征库的key值
    name = get_face_featurelibs_map_key(name)

    # 检查是否包含键 'b'
    if name in FACE_FEATURELIBS_MAP:
        print(name + " 存在于字典中")
        return FACE_FEATURELIBS_MAP.get(name)
    else:
        print(name + " 不存在于字典中")
        return None


# 创建人脸文件夹
def create_featurelibs_face_dir(directory_path,api_endpoint,project_featurelibs_id):
    global FACE_FEATURELIBS_MAP

    post_data = {
        "name":"",
        "parentID":project_featurelibs_id,
        "type":"Face"
    }

    # 遍历文件夹中的文件
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
          
                filename_without_extension = get_face_featurelibs_map_key(filename)
                if get_featurelibs_data(filename_without_extension) == None:
                    post_data["name"] = filename_without_extension
                    print(post_data)
                    # 创建特征库
                    response = requests.post(api_endpoint, json=post_data,headers = {'Authentication': TOKEN})
                    # 处理响应
                    if response.status_code == 200:
                        print(f"文件 {filename_without_extension} 创建成功！")
                    else:
                        print(f"文件 {filename_without_extension} 创建失败，错误码：{response.status_code}")
                        print(f"{api_endpoint} : {response.text}")


def get_face_featurelibs_map_key(name):
    # 去除文件名的后缀
    filename_without_extension = os.path.splitext(name)[0]
    # 将下划线替换为连字符
    filename_without_extension = filename_without_extension.replace('_', '')

    if len(filename_without_extension) > 20:
        filename_without_extension = filename_without_extension[-20:]
    else:
        filename_without_extension = filename_without_extension
    return filename_without_extension


# 获取特征库列表
def fetch_featurelibs_data(api_endpoint):
    global FACE_FEATURELIBS_MAP
    global TOKEN
    response = requests.get(api_endpoint,headers = {'Authentication': TOKEN})
    if response.status_code == 200:
        data = response.json()
        data_list = data["data"]["items"]
        # 将数据列表转换为字典形式，假设数据列表中每个元素都有一个唯一的ID字段
        if data_list:
            FACE_FEATURELIBS_MAP = { item['name']: item for item in data_list}
        else:
            FACE_FEATURELIBS_MAP = {}    
        print("数据列表已更新 SUCCESS")
    else:
        print(f"获取数据列表失败，错误码：{response.status_code}") 
        print(f"{api_endpoint} : {response.text}")
    



TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJVc2VySUQiOiJhZG1pbiIsIkFwcHMiOm51bGwsImV4cCI6NDgzNTIzMTQwNn0.qnAg_sFc2o4hcQslqdM0RYK0mP7K76NCCuNtiHSKlW4"

# 文件夹路径
#folder_path = r'D:\工作交接\穗建人脸考勤\龙口人脸库\test'
folder_path = r'D:\工作交接\穗建人脸考勤\龙口人脸库\d1e394d934ff4db096c3cd681637e432'

project_featurelibs_id="9n65PAueW4mD543aBFv9XTNcx8CWt2Cps9ZmdCiwdxffEzPjDujyZBNGFDguDQhwpWs"

# 指定的API端点
api_endpoint = 'http://192.168.19.110:9090/api/inflet/v1/featurelibs'

list_api=f"http://192.168.19.110:9090/api/inflet/v1/featurelibs?pageNumber=1&pageSize=99999&parentId={project_featurelibs_id}"


if __name__ == "__main__":
    # 获取当前已有的特征库
    fetch_featurelibs_data(list_api)

    # 根据目录，创建对应的特征库
    create_featurelibs_face_dir(folder_path,api_endpoint,project_featurelibs_id)

    # 架子最新的特征库列表
    fetch_featurelibs_data(list_api)

    # 调用函数上传文件
    upload_files(folder_path, api_endpoint)
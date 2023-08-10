import cv2
import colorsys

# FONT = cv2.FONT_HERSHEY_SIMPLEX  # 设置字体类型
FONT = cv2.FONT_ITALIC  # 设置字体类型
FONT_SIZE = 0.5  # 设置字体大小
FONT_COLOR = (0, 0, 0)  # 设置字体颜色为黑色（RGB格式）
FONT_OFFSET = int(6/FONT_SIZE)  # 设置文字偏移量

def is_color_light(rgb):
    r, g, b = rgb
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    return v > 0.5

# 在图像上绘制矩形框
def draw_boxes(image, boxes,color):
    # 在图像上绘制矩形框
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    
    return image

# 绘制标签
def draw_text(image, text, x, y,color):
    global FONT, FONT_SIZE, FONT_COLOR,FONT_OFFSET

    y=y - FONT_OFFSET
    background_color = color

    # 判断是否超出可视区域
    if(y < 40):
        y = y + 40

    # 获取文本尺寸
    (text_width, text_height), _ = cv2.getTextSize(text, FONT, FONT_SIZE, 1)
    text_height = int(text_height * FONT_SIZE)
    cv2.rectangle(image, (int(x), int(y - text_height - FONT_OFFSET)),
              (int(x + text_width + FONT_OFFSET), int(y + FONT_OFFSET)), background_color, -1)

    
    if is_color_light(color):
        FONT_COLOR = (0, 0, 0) # 设置字体颜色为黑色（RGB格式）
    else:
        FONT_COLOR = (255, 255, 255) # 设置字体颜色为白色（RGB格式）    

    # 在图像上绘制文字以及标签背景
    cv2.putText(image, text, (int(x), int(y - 3)), FONT, FONT_SIZE, FONT_COLOR, 1,4)

    return image

# 转成RGB格式
def to_rgb(color):
    return (color[2],color[1],color[0])

def draw_image(image_url,out_url,boxs):

    image = cv2.imread(image_url)
    
    output_image = image.copy()
    for b in boxs:
        bbox=b["bbox"]
        box=bbox["box"]
        x1 = int(box["left_top_x"])
        y1 = int(box["left_top_y"])
        x2 = int(box["right_bottom_x"])
        y2 = int(box["right_bottom_y"])
        color = to_rgb(bbox["color"])
        
        output_image = draw_boxes(output_image, [[x1, y1, x2, y2]],color)

    # 最后画标签，保证标签信息在最上层
    for b in boxs:
        bbox=b["bbox"]
        box=bbox["box"]
        x1 = box["left_top_x"]
        y1 = box["left_top_y"]
        label_text = str(bbox["label"]) + ': ' + str(round(float(bbox["prob"]),2))
        color = to_rgb(bbox["color"])
        output_image = draw_text(output_image, label_text, x1, y1,color)

    # 保存输出图像
    cv2.imwrite(out_url, output_image)    
    return {
        "message" : f"生成推理JPEG图片成功:{ out_url }",
        "out_url" : out_url
    }
# 主函数
if __name__ == "__main__":
    # 原图
    image_url=r".\test_images\0b55556d-632c-48d7-8111-3b7e102697d3.jpg"
    # 输出图片
    out_url=r".\test_images\0b55556d-632c-48d7-8111-3b7e102697d3.jpeg"

    boxs=[{
            "bbox":{
                "box": {
                    "left_top_x": 300, # 左上X坐标
                    "left_top_y": 600,  #  左上Y坐标
                    "right_bottom_x": 400,  # 右下X坐标
                    "right_bottom_y": 800  #  右下Y坐标
                    }, 
                "color": [255, 255, 0, 0],  #  框颜色
                "label": "reflective_vest",  #  标签信息
                "prob": 0.7248  # 置信度      
            },  
        },
        {
            "bbox":{
                "box": {
                    "left_top_x": 138, # 左上X坐标
                    "left_top_y": 296,  #  左上Y坐标
                    "right_bottom_x": 250,  # 右下X坐标
                    "right_bottom_y": 630 #  右下Y坐标
                    },
                "color": [255, 0, 0, 0],  #  框颜色
                "label": "normal_clothes",  #  标签信息
                "prob": 0.7248  # 置信度      
            }
        }
    ]
    
    draw_image(image_url,out_url,boxs)
import PIL

def imagedraw_textsize_c(draw, text, font=None):
    if int(PIL.__version__.split('.')[0]) < 10:
        tw, th = draw.textsize(text, font=font)
    else:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        tw, th = right - left, bottom - top

    return tw, th


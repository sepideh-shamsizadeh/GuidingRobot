

def check_positions(image_positions, laser_positions):
    for x, y in laser_positions:
        for xi, yi, w, h in image_positions:
            if xi <= x <= xi+w:
                if yi <= y < yi+h:
                    print(x, y)
    return 0


if __name__ == '__main__':
    im_p = [[910.0, 252.0, 28.0, 108.0], [451.5, 259.0, 59.0, 168.0]]
    l_p = [[434, 343], [455, 330], [912, 294], [902, 295], [225, 320], [8, 246], [79, 360]]
    check_positions(im_p, l_p)

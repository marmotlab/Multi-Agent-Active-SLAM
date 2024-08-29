from gym.envs.classic_control import rendering #works with gym==0.21.0 installed
import math

#---render world function---
#functions taken from PRIMAL
def create_rectangle(viewer, x, y, width, height, color, empty = False, linewidth = 2, dot = False, permanent = False):
    ps = [(x,y),((x+width),y),((x+width),(y-height)),(x,(y-height))]
    if empty:
        rect = rendering.make_polyline(ps+[ps[0]])
        rect.set_linewidth(linewidth)
        rect.set_color(color[0],color[1],color[2])
    else :
        rect = rendering.FilledPolygon(ps)
        rect.set_color(color[0],color[1],color[2])
    rect.add_attr(rendering.Transform())

    if permanent:
        viewer.add_geom(rect)
    else:
        viewer.add_onetime(rect)

def create_diamond(viewer, x, y, vertical_ray, hor_ray, color, empty = False, linewidth = 2, dot = False, permanent = False):
    ps = [(x, y-vertical_ray), (x+hor_ray, y), (x, y+vertical_ray), (x-hor_ray,y)]
    if empty:
        rect = rendering.make_polyline(ps+[ps[0]])
        rect.set_linewidth(linewidth)
        rect.set_color(color[0],color[1],color[2])
    else :
        rect = rendering.FilledPolygon(ps)
        rect.set_color(color[0],color[1],color[2])
    rect.add_attr(rendering.Transform())

    if permanent:
        viewer.add_geom(rect)
    else:
        viewer.add_onetime(rect)

def draw_circle(viewer, x, y, diameter, size, color, empty = False, linewidth = 2, resolution = 20):
    c = (x+size/2,y-size/2)
    dr = math.pi*2/resolution
    ps = []
    for i in range(resolution):
        x = c[0]+math.cos(i*dr)*diameter/2
        y = c[1]+math.sin(i*dr)*diameter/2
        ps.append((x,y))
    
    if empty:
        circ = rendering.make_polyline(ps+[ps[0]])
        circ.set_linewidth(linewidth)
    else:
        circ = rendering.FilledPolygon(ps)
    
    circ.set_color(color[0],color[1],color[2])
    circ.add_attr(rendering.Transform())

    viewer.add_onetime(circ)

def draw_path(viewer, path, color, linewidth = 3):
    line = rendering.make_polyline(path)
    line.set_linewidth(linewidth)
    line.set_color(color[0],color[1],color[2])
    line.add_attr(rendering.Transform())
    viewer.add_onetime(line)

def draw_arrow(viewer, x, y, dir_x, dir_y, color, linewidth = 10):
    arrow = rendering.Line((x,y), (x+dir_x,y-dir_y))
    arrow.add_attr(rendering.LineWidth(linewidth))
    #arrow.add_attr(rendering.LineStyle(style='curve'))
    arrow.set_color(color[0],color[1],color[2])
    arrow.add_attr(rendering.Transform())
    viewer.add_onetime(arrow)

def split_up_list(list, n_cuts):
    if n_cuts > 0:
        split_list = []
        for i in range(len(list) -1):
            el_1 = list[i]
            el_2 = list[i+1]
            for k in range(n_cuts):
                s_x = k/n_cuts*(el_2[0]-el_1[0]) + el_1[0]
                s_y = k/n_cuts*(el_2[1]-el_1[1]) + el_1[1]
                split_list.append((s_x, s_y))
        split_list.append(list[-1])
        return split_list

def dot_list(list, ratio):
    list_of_list = []
    for i in range(len(list) -1):
        if i % ((len(list_of_list)//10 - i//10) +1)//(int(1/ratio)) == 0:
            list_of_list.append([list[i], list[i+1]])
    return list_of_list

def draw_trace(viewer, line, color, n_cuts = 1, ratio = 1, linewidth = 1):
    split_line = split_up_list(line, n_cuts)
    list_of_lines = dot_list(split_line, ratio)
    for portion in list_of_lines:
        dot = rendering.make_polyline(portion)
        dot.set_linewidth(linewidth)
        dot.set_color(color[0],color[1],color[2])
        dot.add_attr(rendering.Transform())
        viewer.add_onetime(dot)
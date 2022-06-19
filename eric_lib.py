import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import random
import numpy as np, pygame
pygame.init()


# Color

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

LIGHT_GREY = (192, 192, 192)
GREY = (128, 128, 128)
DARK_GREY = (64, 64, 64)

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

YELLOW = (255, 255, 0)
PURPLE = (255, 0, 255)
CYAN = (0, 255, 255)

DEBUG_COLOR = RED

def random_color():
    return tuple(
        random.randint(0, 255) for i in range(3)
    )

def colored_surface(color, size=(1, 1)):
    surface = pygame.Surface(size)
    surface.fill(color)
    return surface


# Basic Graphics

def grid_lines(dimensions, box_size, offset=(0, 0)):
    lines = []

    for x in range(dimensions[0] + 1):
        lines.append((
            (
                offset[0] + x * box_size[0],
                offset[1]
            ),
            (
                offset[0] + x * box_size[0],
                offset[1] + dimensions[1] * box_size[1]
            )
        ))

    for y in range(dimensions[1] + 1):
        lines.append((
            (
                offset[0],
                offset[1] + y * box_size[1]
            ),
            (
                offset[0] + dimensions[0] * box_size[0],
                offset[1] + y * box_size[1],
            )
        ))

    return lines

def render_text(message, size=20, font=None,
    bold=False, italic=False,
    color=BLACK, background=None,
    antialias=False
):
    if not font:
        font = pygame.font.Font(None, size)
    else:
        font = pygame.font.SysFont(font, size, bold, italic)
    
    return font.render(str(message), True, color, background)

ALIGN_LEFT = -1
ALIGN_TOP = -1
ALIGN_CENTER = 0
ALIGN_RIGHT = 1
ALIGN_BOTTOM = 1
def aligned_blit(
    surface, render, pos, x_align, y_align
):
    rect = render.get_rect()

    if x_align == ALIGN_LEFT:
        rect.left = pos[0]
    elif x_align == ALIGN_CENTER:
        rect.centerx = pos[0]
    elif x_align == ALIGN_RIGHT:
        rect.right = pos[0]
    
    if y_align == ALIGN_TOP:
        rect.top = pos[1]
    elif y_align == ALIGN_CENTER:
        rect.centery = pos[1]
    elif y_align == ALIGN_BOTTOM:
        rect.bottom = pos[1]

    surface.blit(render, rect)


# Math

def vector(p=(0, 0)):
    return np.array(p, np.float64)

def point(p=(0, 0)):
    return tuple([float(i) for i in p])

def int_to_bits(n:int, length:int) -> tuple:
    n %= 2**length
    
    bits = []
    for i in range(length-1, -1, -1):
        value = 2**i
        if n // value > 0:
            n -= value

            bits.append(1)
        else:
            bits.append(0)
    return tuple(bits)

def within(x, min, max):
    "inclusive minimum and maximum"
    assert min < max
    if x >= min and x <= max:
        return True
    else:
        return False

def constrain(x, min, max):
    "inclusive minimum and maximum"
    assert min <= max
    if x < min:
        return min
    elif x > max:
        return max
    else:
        return x

def lerp(t, p1, p2):
    p1 = vector(p1)
    p2 = vector(p2)

    return (1 - t)*p1 + t*p2

def bezier(t, points):
    if len(points) <= 1:
        return points[0]

    new = []
    for i in range(len(points)-1):
        new.append(lerp(t, points[i], points[i+1]))

    return bezier(t, new)

def random_point(min, max):
    point = []
    for x in zip(min, max):
        point.append(random.uniform(*x))
    return point

def distance(p1, p2=(0, 0)):
    p1 = vector(p1)
    return np.linalg.norm(p1 - p2)

def normalize(v):
    m = np.linalg.norm(v)
    if m:
        return vector(v) / m
    else: return vector((0, 0))

def manhattan_distance(p1, p2):
    p1 = vector(p1)
    v = p1 - p2
    return abs(v[0]) + abs(v[1])

def rotate(point, theta, pivot=(0, 0)):
    c = np.cos(theta)
    s = np.sin(theta)

    x, y = vector(point) - pivot
    p = vector((x*c - y*s, x*s + y*c))
    return p + pivot

def angle_to_vector(theta):
    return vector((np.cos(theta), np.sin(theta)))

def compare_points(p1, p2):
    for a, b in zip(p1, p2):
        if a != b:
            return False
    return True

class Box:
    def __init__(self, pos, size):
        self.pos = vector(pos)
        self.size = vector(size)

        assert self.pos.size == 2
        assert self.size.size == 2
    
    def __getattr__(self, name):
        # All defined attributes are handled automatically

        if name == "x":
            return self.pos[0]
        elif name == "y":
            return self.pos[1]
        elif name == "w":
            return self.size[0]
        elif name == "h":
            return self.size[1]

        elif name == "left":
            return self.x
        elif name == "bottom":
            # return self.y
            return self.y+self.h
        elif name == "right":
            return self.x+self.w
        elif name == "top":
            # return self.y+self.h
            return self.y

        elif name == "center":
            return vector((
                self.x+self.w/2,
                self.y+self.h/2
            ))

        else:
            raise AttributeError(self.__class__, name)

    def __setattr__(self, name, value):
        if name == "x":
            self.pos[0] = value
        elif name == "y":
            self.pos[1] = value
        elif name == "w":
            self.size[0] = value
        elif name == "h":
            self.size[1] = value

        elif name == "left":
            self.x = value
        elif name == "bottom":
            # self.y = value
            self.y = value-self.h
        elif name == "right":
            self.x = value-self.w
        elif name == "top":
            # self.y = value-self.h
            self.y = value

        elif name == "center":
            self.x = value[0]-self.w/2
            self.y = value[1]-self.h/2

        else:
            # self.__dict__[name] = value
            super().__setattr__(name, value)
    
    def __contains__(self, item):
        # if isinstance
        return self.point_within(item)
    
    def to_pygame_Rect(self):
        return pygame.Rect(self.pos, self.size)
    
    def get_points(self):
        return (
            vector((self.left, self.top)),
            vector((self.right, self.top)),
            vector((self.right, self.bottom)),
            vector((self.left, self.bottom))
        )
    
    def get_bounding_segments(self):
        pts = self.get_points()
        return (
            (pts[0], pts[1]),
            (pts[1], pts[2]),
            (pts[2], pts[3]),
            (pts[3], pts[0])
        )
    
    def point_within(self, p):
        if not within(p[0], self.left, self.right):
            return False
        if not within(p[1], self.top, self.bottom):
            return False
        return True
    
    def constrain_point(self, p):
        return vector((
            constrain(p[0], self.left, self.right),
            constrain(p[1], self.top, self.bottom)
        ))
    
    def constrain_to(self, box):
        assert self.size[0] <= box.size[0]
        assert self.size[1] <= box.size[1]

        # Find the Box the top-left point of self could be in
        p_box = Box(
            box.pos,
            box.size - self.size
        )
        
        self.pos = p_box.constrain_point(self.pos)

def two_point_box(p1, p2):
    min_point = (min(p1[0], p2[0]), min(p1[1], p2[1]))
    max_point = (max(p1[0], p2[0]), max(p1[1], p2[1]))
    return Box(
        min_point,
        vector(max_point) - min_point
    )

def containing_box(points=[], boxes=[]):
    points = points.copy()
    for b in boxes:
        points.extend(b.get_points())
    
    assert len(points) > 0
    
    zipped = tuple(zip(points))

    min_point = (
        min(zipped[0]),
        min(zipped[1])
    )
    max_point = (
        max(zipped[0]),
        max(zipped[1])
    )

    return two_point_box(min_point, max_point)

class Line:
    def __init__(self, p1, p2, t_max=1, t_min=0):
        if t_min is None: t_min = np.NINF
        if t_max is None: t_max = np.PINF

        self.p1 = vector(p1)
        self.p2 = vector(p2)
        self.t_min = t_min
        self.t_max = t_max

        assert self.p1.size == 2, (self.p1)
        assert self.p2.size == 2, (self.p2)
    
    def get_coefficients(self):
        return [
            (a, b - a) for a, b in zip(self.p1, self.p2)
        ]
    
    def t_in(self, t):
        return within(t, self.t_min, self.t_max)

    def at_t(self, t):
        if not self.t_in(t): return
        
        return [
            (a+b*t) for a, b in self.get_coefficients()
        ]
    
    def intersect_t1_t2(self, other):
        co1 = self.get_coefficients()
        co2 = other.get_coefficients()

        a1, b1 = co1[0]
        c1, d1 = co1[1]
        a2, b2 = co2[0]
        c2, d2 = co2[1]

        try:
            t1 = float(b2*c1-b2*c2-a1*d2+a2*d2)/float(b1*d2-b2*d1)
            t2 = float(a1-a2+b1*t1)/float(b2)
        except ZeroDivisionError:
            t2 = float(b1*c2-b1*c1-a2*d1+a1*d1)/float(b2*d1-b1*d2)
            t1 = float(a2-a1+b2*t2)/float(b1)
        # except smh

        return t1, t2
    
    def intersect(self, other):
        t1, t2 = self.intersect_t1_t2(other)

        if self.t_in(t1) and other.t_in(t2):
            return self.at_t(t1)
        else:
            return


# Simulation

class SimulationHandle:
    def __init__(self, size):
        self.size = vector(size)
        self.area = Box((0, 0), self.size)
        
        self.categories = {}
        self.add_category("DEBUG")
        self.background = BLACK
        self.mouse_offset = vector((0, 0))

        self.ticks = -1
    
    def add_category(self, key):
        if key in self.categories:
            raise KeyError(key)
        else:
            self.categories[key] = []
    
    def update(self):
        self.ticks += 1

        events = pygame.event.get()
        keys_pressed = pygame.key.get_pressed()

        for e in events:
            if e.type in (
                pygame.MOUSEMOTION,
                pygame.MOUSEBUTTONDOWN,
                pygame.MOUSEBUTTONUP
            ):
                # self.event(
                #     e,
                #     vector(e.pos) - self.mouse_offset,
                #     keys_pressed
                # )
                e.pos = vector(e.pos) - self.mouse_offset
            
            # else:
                # self.event(e, None, keys_pressed)

        for c in self.categories.values():
            i = 0
            while i < len(c):
                updatable = c[i]

                updatable.update(events, keys_pressed)

                if updatable.invalid:
                    c.pop(i)
                else:
                    i += 1
    
    def render(self):
        surface = pygame.Surface(self.size)
        surface.fill(self.background)

        for c in reversed(self.categories.values()):
            for updatable in c:
                updatable.render_to(surface)
        
        return surface

class Updatable:
    def __init__(self, handle, timeout=None):
        self.handle = handle
        self.invalid = False

        self.debug_color = DEBUG_COLOR
        self.expiry = np.PINF
    
    def set_expiration(self, ticks=1):
        self.expiry = self.handle.ticks + ticks
    
    def update(self, events, keys_pressed):
        if self.expiry <= self.handle.ticks:
            self.invalid = True
    
    def render_to(self, surface):
        raise NotImplementedError

# class UpdatableScript(Updatable):
#     def render_to(self, surface): pass

class UpdatableBox(Updatable, Box):
    def __init__(self, handle, pos, size):
        Updatable.__init__(self, handle)
        Box.__init__(self, pos, size)

class UpdatableLine(Updatable, Line):
    def __init__(
        self, handle, p1, p2, t_max=None, t_min=None
    ):
        Updatable.__init__(self, handle)
        Line.__init__(self, p1, p2, t_max=None, t_min=None)
    
    def render_to(self, surface):
        pygame.draw.line(
            surface, self.debug_color,
            self.p1, self.p2
        )

class DebugLine(UpdatableLine):
    def __init__(self, handle, p1, p2, t_max=None, t_min=None):
        super().__init__(handle, p1, p2, t_max=None, t_min=None)
        self.set_expiration(1)


# Advanced Graphics

class RenderModel:
    def __init__(
        self, renders:tuple, name=None, offset=(0, 0)
    ):
        self.renders = renders
        self.default_render = self.renders[0]

        for i, r in enumerate(self.renders):
            assert isinstance(r, pygame.Surface), (i, r)

        self.name = name
        self.offset = offset
        self.size = self.default_render.get_rect().size
        self.box = Box(self.offset, self.size)

        self.events_handler = lambda instance, events, keys_pressed: None
    
    def render(self, state=0, size=None):
        if size:
            return pygame.transform.smoothscale(
                self.render, size
            )
        else:
            return self.renders[state]
    
    def render_to(self, surface, pos, state=0, size=None):
        surface.blit(self.render(state, size), pos)

class ParentRenderModel(RenderModel):
    def __init__(self, render, *extra_renders, name=None, offset=(0, 0)):
    #, child_models=[], offset=(0, 0)):
        super().__init__(render, *extra_renders, name=name, offset=offset)

        # self.child_models = child_models
    
    # def render(self, state=0, size=None, **child_states):
    #     if not size:
    #         size = self.box.size

    #     bounds = containing_box(boxes=[
    #         c.box for c in self.child_models
    #     ]+[self.box])

    #     surface = pygame.Surface(size)

    #     surface.blit(
    #         self.renders[state],
    #         -bounds.pos
    #     )
    #     for i in range(len(self.child_models)):
    #         c = self.child_models[i]
    #         pos = c.offset-bounds.pos
    #         state = child_states[i]

    #         if c in child_states:
    #             c.render_to(surface, pos, state)
    #         else:
    #             c.render_to(surface, pos)
        
    #     return pygame.transform.smoothscale(
    #         surface, size
    #     )
    
    def render_to(self, surface, pos, state=0, size=None):#, **child_states):
        surface.blit(
            self.render(state, size),#, **child_states),
            pos
        )

class RenderInstance(UpdatableBox):
    def __init__(self, handle, pos, model:RenderModel):
        super().__init__(handle, pos, model.size)

        self.model = model
        self.render_state = 0

    def render_to(self, surface):
        self.model.render_to(
            surface, self.pos, self.render_state
        )
    
    def update(self, events, keys_pressed):
        super().update(events, keys_pressed)
        
        self.model.events_handler(
            self, events, keys_pressed
        )
    
    def click(self, pos):
        self.on_click(pos)
    
    def on_click(self, pos):
        pass

class ParentRenderInstance(RenderInstance):
    def __init__(self, handle, pos, model):
        super().__init__(handle, pos, model)

        self.children = []

    def __setattr__(self, name, value):
        if name == "pos" and hasattr(self, "pos"):
            dif = value - self.pos
            for c in self.children:
                c.pos += dif
        
        elif name == "invalid" and hasattr(self, "invalid"):
            for c in self.children:
                c.invalid = self.invalid

        super().__setattr__(name, value)
    
    def render_to(self, surface):
        super().render_to(surface)
        for c in self.children:
            c.render_to(surface)
    
    def click(self, pos):
        super().click(pos)
        for c in self.children:
            if pos in c:
                c.on_click()
                return


# User input

WASD_VECTORS = (
    vector((0.0, -1.0)),
    vector((-1.0, 0.0)),
    vector((0.0, 1.0)),
    vector((1.0, 0.0))
)

WASD_KEYS = (
    pygame.K_w,
    pygame.K_a,
    pygame.K_s,
    pygame.K_d
)

def wasd_modified_vectors(w=1, a=1, s=1, d=1):
    return (
        WASD_VECTORS[0] * w,
        WASD_VECTORS[1] * a,
        WASD_VECTORS[2] * s,
        WASD_VECTORS[3] * d
    )

def wasd_events_handler(
    instance, events, keys_pressed=[], speeds=(1, 1, 1, 1), velocity=False
):
    vectors = wasd_modified_vectors(*speeds)
    for k, v in zip(WASD_KEYS, vectors):
        if keys_pressed[k]:
            if velocity:
                instance.velocity += v
            else:
                instance.pos += v


# Interface

class ButtonModel(RenderModel):
    pass

class Button(RenderInstance):
    def __init__(self, handle, pos, model, function):
        super().__init__(handle, pos, model)

        self.function = function

    def on_click(self, pos):
        self.function(pos)

class MenuModel(ParentRenderModel):
    pass

class Menu(ParentRenderInstance):
    # def click(self, pos):
    #     local = pos - self.pos
    #     for c in self.children:
    #         if local in c:
    #             c.click()
    #             break

    pass


# Applications

class Drag_Drop_Area(SimulationHandle):
    def __init__(self, size, toolbox_zone):
        super().__init__(size)

        self.categories["MENUS"] = []
        self.categories["BUTTONS"] = []
        self.categories["OBJECTS"] = []

        self.clickable_categories = (
            (
                "BUTTONS",
                lambda x, pos: x.click(pos)
            ),
            (
                "OBJECTS",
                lambda x, pos: self.select(x, pos)
            )
        )

        self.selected = None
        self.selected_offset = None

        # self.toolbox_zone = toolbox_zone
        self.toolbox = Menu(
            self, toolbox_zone.pos, MenuModel(
                colored_surface(
                    WHITE, toolbox_zone.size
                ), name="TOOLBOX"
            )
        )
        self.categories["MENUS"].append(self.toolbox)

    def event(self, e, pos, keys_pressed):
        if e.type == pygame.MOUSEBUTTONDOWN:
            if e.button == 1:
                self.click(pos)
        
        elif e.type == pygame.MOUSEMOTION:
            self.update_selected(pos)

    def click(self, pos):
        if self.selected:
            self.stop_select(pos)
        else:
            return self.start_select(pos)
    
    def find_clicked(self, pos, category):
        for updatable in self.categories[category]:
            if pos in updatable:
                return updatable
    
    def select(self, updatable, mouse_pos):
        self.selected = updatable
        self.selected_offset = updatable.pos-mouse_pos
    
    def start_select(self, pos):
        for c, f in self.clickable_categories:
            x = self.find_clicked(pos, c)
            if x:
                f(x, pos)
                return
    
    def stop_select(self, pos):
        if pos in self.toolbox:
            self.selected.invalid = True
        else:
            self.update_selected(pos)
        self.selected = None
        self.selected_offset = None
    
    def update_selected(self, mouse_pos):
        if self.selected:
            self.selected.pos = mouse_pos+self.selected_offset
            self.selected.constrain_to(self.area)


# Mazes

class Node:
    def __init__(self, pos):
        self.pos = vector(pos)
        self.connections = []

def get_adjacent_tiles(pos):
    pos = vector(pos)
    return [
        pos + (1, 0),
        pos + (0, 1),
        pos + (-1, 0),
        pos + (0, -1)
    ]

def get_adjacent_tile_corners(pos):
    pos = vector(pos)
    return [
        pos + (1, 1),
        pos + (-1, 1),
        pos + (-1, -1),
        pos + (1, -1)
    ]

def tile_grid(dimensions, filler=None):
    grid = []
    for x in range(dimensions[0]):
        grid.append([])
        for y in range(dimensions[1]):
            grid[x].append(filler)
    return grid

def create_maze_node_grid(dimensions):
    bounds = Box((0, 0), dimensions)
    start = vector(tuple((random.randint(0, i-1) for i in dimensions)))
    
    node_grid = tile_grid(dimensions)
    
    start_node = Node(start)
    node_stack = [start_node]
    node_grid[start[0]][start[1]] = start_node

    while node_stack:
        possible_locations = get_adjacent_tiles(
            node_stack[-1].pos
        )
        success = False
        while possible_locations:
            pos = possible_locations.pop(random.randint(
                0, len(possible_locations)-1
            ))

            if not pos in bounds:
                continue
            elif node_grid[pos[0]][pos[1]]:
                continue
            else:
                node = Node(pos)
                node.connections.append(node_stack[-1])
                node_stack.append(node)
                node_grid[pos[0]][pos[1]] = node
                
                success = True
                break
        if not success:
            node_stack.pop()
    
    return node_grid

def create_maze(dimensions, box_size, offset=(0, 0)):
    maze = create_maze_node_grid(dimensions)
    dimensions = vector(dimensions)
    box_size = vector(box_size)
    bounds = Box(
        offset, dimensions*box_size
    )

    paths = []
    for x in range(dimensions[0]*2 - 1):
        if x%2 == 0:
            paths.append([
                False for y in range(dimensions[1]-1)
            ])
        else:
            paths.append([
                False for y in range(dimensions[1])
            ])
    for x, column in enumerate(maze):
        for y, node in enumerate(column):
            if node:
                for connection in node.connections:
                    dif = tuple(
                        connection.pos - node.pos
                    )
                    if dif == (1, 0):
                        paths[x*2 + 1][y] = True
                    elif dif == (0, 1):
                        paths[x*2][y] = True
                    elif dif == (-1, 0):
                        paths[x*2 - 1][y] = True
                    elif dif == (0, -1):
                        paths[x*2][y-1] = True

                    else:
                        raise AssertionError
    
    proto_segments = []
    for x, column in enumerate(paths):
        proto_segments.append([])
        for y, seg in enumerate(column):
            proto_segments[x].append(not seg)
    
    wonky_segments = []
    for x, column in enumerate(proto_segments):
        for y, seg in enumerate(column):
            if seg:
                p1 = vector((x//2, y))*box_size + offset

                if x%2 == 0:
                    wonky_segments.append((
                        p1,
                        (
                            p1[0], p1[1] + box_size[1]
                        )
                    ))
                else:
                    wonky_segments.append((
                        p1,
                        (
                            p1[0] + box_size[0], p1[1]
                        )
                    ))
    
    walls = list(bounds.get_bounding_segments())
    for l in wonky_segments:
        l += box_size/2
        center = lerp(0.5, l[0], l[1])
        walls.append((
            rotate(l[0], np.pi/2, center),
            rotate(l[1], np.pi/2, center)
        ))

    return walls

def create_tile_maze(dimensions, room_size=(1, 1)):
    tiles = tile_grid((
        dimensions[0]*room_size[0]*2 + 1,
        dimensions[1]*room_size[1]*2 + 1
    ), False)
    
    wall_segments = create_maze(
        dimensions, vector(room_size) + 1
    )

    for w in wall_segments:
        bounds = two_point_box(*w)
        for x in range(
            int(bounds.left),
            int(bounds.right)+1
        ):
            for y in range(
                int(bounds.top),
                int(bounds.bottom)+1
            ):
                tiles[x][y] = True

    return tiles


# Pathfinding

# I should learn about fibonacci heaps. - From Dad

class PathNode(Node):
    def __init__(self, pos):
        super().__init__(pos)

        self.f = None
        self.g = None
        self.h = None
    
    def to_list(self):
        node = self
        path = [node]
        while node.connections:
            node = node.connections[0]
            path.append(node)
        return path

def pathfind_a_star_tiles(
    start, goal, tiles
):
    if tiles[goal[0]][goal[1]]:
        return False

    start_node = PathNode(start)
    start_node.g = 0
    start_node.f = 0
    
    open = [start_node]
    closed = []

    while open:
        # Find the best node
        q = open[0]
        for i, n in enumerate(open):
            if not i:
                continue
            
            if n.f < q.f:
                q = n
        
        # Remove the best node
        open.remove(q)
        
        # Get succcessors
        for p in get_adjacent_tiles(q.pos):
            # Check to see if tile is blocked
            if tiles[p[0]][p[1]]:
                continue

            # Create node
            successor = PathNode(p)
            successor.connections.append(q)

            if compare_points(p, goal):
                # closed.append(q)
                # closed.append(successor)
                return successor.to_list()

            # Calculate cost
            successor.g = q.g + 1
            successor.h = manhattan_distance(p, goal)
            successor.f = successor.g + successor.h

            # Check open and closed lists
            valid = True
            for n in open + closed:
                if compare_points(
                    successor.pos, n.pos
                ):
                    if successor.f >= n.f:
                        valid = False
                        break

            # Add to open list
            if valid:
                open.append(successor)
        
        # Add to closed list
        closed.append(q)
    
    return False


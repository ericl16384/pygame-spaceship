import eric_lib as lib, numpy as np, pygame


def wide_line(p1, p2, width):
    p1 = lib.vector(p1)
    p2 = lib.vector(p2)

    v = p2 - p1
    p = lib.rotate(v, np.pi/2)
    n = lib.normalize(p) * (width/2)

    return (
        p1 + n,
        p1 - n,
        p2 - n,
        p2 + n
    )

def triangle_line(p1, p2, width):
    p1 = lib.vector(p1)
    p2 = lib.vector(p2)

    v = p2 - p1
    p = lib.rotate(v, np.pi/2)
    n = lib.normalize(p) * (width/2)

    return (
        p1 + n,
        p1 - n,
        p2
    )


class Thruster:
    def __init__(self, pos, angle):
        self.pos = lib.vector(pos)
        self.angle = angle

        self.thrust = 0
        self.max_thrust = 100

        self.init_constants()

        self.vertices = (
            (-8, 3),
            (-8, -3),
            (8, -1),
            (8, 1)
        )
        self.thrust_plume_start = (-10, 0)
        self.thrust_plume_width = 4
        self.thrust_plume_scale = 1
    
    def init_constants(self):
        pass

    def render_to(self, surface, transform, body=True, plume=True):
        t = lambda p:transform(
            self.to_ship_space(p)
        )
        ts = lambda pts:[t(p) for p in pts]

        # Body
        if body:
            pygame.draw.polygon(
                surface, lib.YELLOW,
                ts(self.vertices)
            )

        # Plume
        if plume:
            plume_start = self.thrust_plume_start
            pygame.draw.polygon(
                surface, lib.RED, ts(triangle_line(
                    plume_start,
                    plume_start - self.get_local_thrust_vector()*self.thrust_plume_scale,
                    self.thrust_plume_width
                ))
            )

    def to_ship_rotation(self, v):
        return lib.rotate(v, -self.angle)
    
    def to_local_rotation(self, v):
        return lib.rotate(v, self.angle)
    
    def to_ship_space(self, point):
        return self.to_ship_rotation(point) + self.pos
    
    def contrain_thrust(self):
        self.thrust = lib.constrain(
            self.thrust, 0, self.max_thrust
        )
    
    def get_local_thrust_vector(self):
        return lib.vector((self.thrust, 0))
    
    def get_ship_thrust_vector(self):
        return self.to_ship_rotation(
            self.get_local_thrust_vector()
        )
    
class Ship(lib.Updatable):
    def __init__(self, handle, pos, angle, thrusters):
        super().__init__(handle)
        self.pos = lib.vector(pos)
        self.angle = angle
        self.thrusters = list(thrusters)

        self.velocity = lib.vector((0, 0))
        self.angular_velocity = 0

        self.mass = 1000
        self.angular_mass = 100000

        self.init_constants()

        # Jedi starfighter
        self.hull_vertices = (
            (52.83018867924528, -3.2349160732194234e-15),
            (-13.207547169811319, 24.90566037735849),
            (-24.528301886792455, 22.830188679245285),
            (-28.49056603773585, 12.07547169811321),
            (-51.320754716981135, 3.142489899698869e-15),
            (-32.264150943396224, -12.264150943396226),
            (-28.49056603773585, -12.641509433962263),
            (-24.71698113207547, -22.830188679245285),
            (-13.396226415094342, -25.09433962264151)
        )
    
    def init_constants(self):
        # self.max_x = 0
        # self.max_y = 0
        # self.max_yaw = 0

        # for t in self.thrusters:
        #     self.max_x += abs(t.direction_vector[0])
        #     self.max_y += abs(t.direction_vector[1])
        #     self.max_yaw += abs(t.rotation_scalar)
        pass
    
    def update(self, events, keys_pressed):
        super().update(events, keys_pressed)

        for t in self.thrusters:
            self.apply_force(
                t.get_ship_thrust_vector(),
                t.pos, True
            )

        self.pos += self.velocity
        self.angle += self.angular_velocity
        self.angle %= np.pi*2

        self.command_thrusters(events, keys_pressed)

        for t in self.thrusters:
            t.contrain_thrust()
    
    def render_to(self, surface):
        # Thrusters
        for t in self.thrusters:
            t.render_to(surface, self.to_global_space, plume=False)

        # Hull
        pygame.draw.polygon(
            surface, lib.BLUE,
            [
                self.to_global_space(p)
                # self.render_transform(p)
                for p in self.hull_vertices
            ]
        )

        # Thrusters
        for t in self.thrusters:
            t.render_to(surface, self.to_global_space, body=False)
        
        # # Center (debug)
        # pygame.draw.circle(
        #     surface, lib.RED, self.pos, 5
        # )
        
        # # Velocity (debug)
        # pygame.draw.aaline(
        #     surface, lib.GREEN, self.pos,
        #     self.pos + self.velocity*50
        # )
    
    def command_thrusters(self, events, keys_pressed):
        w = keys_pressed[pygame.K_w]
        a = keys_pressed[pygame.K_a]
        s = keys_pressed[pygame.K_s]
        d = keys_pressed[pygame.K_d]
        f = keys_pressed[pygame.K_UP]
        l = keys_pressed[pygame.K_LEFT]
        o = keys_pressed[pygame.K_DOWN]
        r = keys_pressed[pygame.K_RIGHT]
        p = keys_pressed[pygame.K_SPACE]

        self.thrusters[0].thrust = w * 50
        self.thrusters[1].thrust = w * 50
        self.thrusters[2].thrust = d * 10
        self.thrusters[3].thrust = a * 10

        if p:
            self.pos = lib.vector((200, 200))
            self.stop_movement()

        # x = 0
        # if keys_pressed[pygame.K_d]:
        #     x += 1
        # if keys_pressed[pygame.K_a]:
        #     x -= 1
        # y = 0
        # if keys_pressed[pygame.K_s]:
        #     y += 1
        # if keys_pressed[pygame.K_w]:
        #     y -= 1
        
        # yaw = 0
        # if keys_pressed[pygame.K_LEFT]:
        #     yaw += 1
        # if keys_pressed[pygame.K_RIGHT]:
        #     yaw -= 1
        
        # if keys_pressed[pygame.K_SPACE]:
        #     self.stop_movement()

        # x, y = lib.normalize((x, y))
    
    def apply_force(self, v, pos=None, local=False):
        if pos is None:
            tangential = 0
        else:
            # https://byjus.com/tangential-velocity-formula/
            radius = lib.rotate(pos, -np.pi/2)
            # radius = lib.normalize(radius)
            tangential = np.dot(radius, v)

        if local:
            v = self.to_global_rotation(v)
            # pos = self.to_global_rotation(pos)
        else:
            v = lib.vector(v)
            # pos = lib.vector(pos)
        
        self.velocity += v / self.mass
        self.angular_velocity += tangential / self.angular_mass

    def to_global_rotation(self, v):
        return lib.rotate(v, -self.angle)
        # return lib.rotate(v, self.angle)
    
    def to_local_rotation(self, v):
        return lib.rotate(v, self.angle)
        # return lib.rotate(v, -self.angle)
    
    def to_global_space(self, point):
        return self.to_global_rotation(point) + self.pos
    
    def stop_movement(self):
        self.velocity = lib.vector((0, 0))
        self.angular_velocity = 0
    

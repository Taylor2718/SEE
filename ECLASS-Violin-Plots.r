##E-CLASS UROP Summer Project
##Violin plots for E-CLASS YOU score and Expert Score.

library(dplyr)
library(ggplot2)

df_tot <- read.csv("df_processed_conv.csv")

geom_flat_violin_right <- function(mapping = NULL, data = NULL, stat = "ydensity",
                        position = "dodge", trim = TRUE, scale = "area",
                        show.legend = NA, inherit.aes = TRUE, ...) {
  layer(
    data = data,
    mapping = mapping,
    stat = stat,
    geom = GeomFlatViolinRight,
    position = position,
    show.legend = show.legend,
    inherit.aes = inherit.aes,
    params = list(
      trim = trim,
      scale = scale,
      ...
    )
  )
}

geom_flat_violin_left <- function(mapping = NULL, data = NULL, stat = "ydensity",
                        position = "dodge", trim = TRUE, scale = "area",
                        show.legend = NA, inherit.aes = TRUE, ...) {
  layer(
    data = data,
    mapping = mapping,
    stat = stat,
    geom = GeomFlatViolinLeft,
    position = position,
    show.legend = show.legend,
    inherit.aes = inherit.aes,
    params = list(
      trim = trim,
      scale = scale,
      ...
    )
  )
}
GeomFlatViolinRight <-
  ggproto("GeomFlatViolin", Geom,
          setup_data = function(data, params) {
            data$width <- data$width %||%
              params$width %||% (resolution(data$x, FALSE) * 0.9)

            # ymin, ymax, xmin, and xmax define the bounding rectangle for each group
            data %>%
              group_by(group) %>%
              mutate(ymin = min(y),
                     ymax = max(y),
                     xmin = x - width / 2,
                     xmax = x)
          },

          draw_group = function(data, panel_scales, coord) {
            # Find the points for the line to go all the way around
            data <- transform(data, 
                              xmaxv = x,
                              xminv = x + violinwidth * (-xmin + x))

            # Make sure it's sorted properly to draw the outline
            newdata <- rbind(plyr::arrange(transform(data, x = xminv), y),
                             plyr::arrange(transform(data, x = xmaxv), -y))

            # Close the polygon: set first and last point the same
            # Needed for coord_polar and such
            newdata <- rbind(newdata, newdata[1,])

            ggplot2:::ggname("geom_flat_violin", GeomPolygon$draw_panel(newdata, panel_scales, coord))
          },

          draw_key = draw_key_polygon,

          default_aes = aes(weight = 1, colour = "grey20", fill = "white", size = 0.5,
                            alpha = NA, linetype = "solid"),

          required_aes = c("x", "y")
)
GeomFlatViolinLeft <-
  ggproto("GeomFlatViolin", Geom,
          setup_data = function(data, params) {
            data$width <- data$width %||%
              params$width %||% (resolution(data$x, FALSE) * 0.9)

            # ymin, ymax, xmin, and xmax define the bounding rectangle for each group
            data %>%
              group_by(group) %>%
              mutate(ymin = min(y),
                     ymax = max(y),
                     xmin = x - width / 2,
                     xmax = x)
          },

          draw_group = function(data, panel_scales, coord) {
            # Find the points for the line to go all the way around
            data <- transform(data, 
                              xmaxv = x,
                              xminv = x + violinwidth * (xmin - x))

            # Make sure it's sorted properly to draw the outline
            newdata <- rbind(plyr::arrange(transform(data, x = xminv), y),
                             plyr::arrange(transform(data, x = xmaxv), -y))

            # Close the polygon: set first and last point the same
            # Needed for coord_polar and such
            newdata <- rbind(newdata, newdata[1,])

            ggplot2:::ggname("geom_flat_violin", GeomPolygon$draw_panel(newdata, panel_scales, coord))
          },

          draw_key = draw_key_polygon,

          default_aes = aes(weight = 1, colour = "grey20", fill = "white", size = 0.5,
                            alpha = NA, linetype = "solid"),

          required_aes = c("x", "y")
)

theme <-   theme(
    axis.ticks.x = element_blank(),
    axis.title.x = element_blank(),
    axis.line = element_line(colour = "grey50"),
    panel.grid = element_line(color = "#b4aea9"),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(linetype = "dashed"),
    panel.background = element_rect(fill = "#ffffff", color = "#ffffff"),
    plot.background = element_rect(fill = "#ffffff", color = "#ffffff"),
    text = element_text(size = 20)
  )
a_b_plot<-
        ggplot(df_tot) + geom_flat_violin_left(aes(x=1,y=a_pre_sum),color="blue",fill="blue")+ geom_flat_violin_right(aes(x=1,y=a_post_sum),color="#fc7d7d",fill="#fc7d7d") +
        geom_flat_violin_left(aes(x=2.1,y=b_pre_sum),color="blue",fill="blue")+ geom_flat_violin_right(aes(x=2.1,y=b_post_sum),color="#fc7d7d",fill="#fc7d7d") +
        ylim(-30, 30) + xlim(0.5,2.6) +  #geom_boxplot(aes(x=1,y=a_pre_sum),color="blue",width=0.1) #geom_boxplot(aes(x=2,y=a_post_sum),color="red",width=0.1)
        geom_segment(aes(x=0.5,xend=1,y=mean(a_pre_sum),yend=mean(a_pre_sum))) +
        geom_segment(aes(x=1,xend=1.5,y=mean(a_post_sum),yend=mean(a_post_sum))) +
        geom_segment(aes(x=1.6,xend=2.1,y=mean(b_pre_sum),yend=mean(b_pre_sum))) +
        geom_segment(aes(x=2.1,xend=2.6,y=mean(b_post_sum),yend=mean(b_post_sum))) +
        labs(y = "E-CLASS Score") + scale_y_continuous(breaks=seq(-30,45,10)) +
        scale_x_continuous(breaks=c(1,2.1),labels=c("1" = "YOU Item", "2.1" = "Expert Item")) + theme
a_b_plot
ggsave(
  filename = "R-Figures/tot_ab_violin.png",
  plot = a_b_plot,
  width = 8,
  height = 8,
  device = "png"
)
package com.kaleidofin;

import java.util.Comparator;

public class Segment {
	
	int x;
	int y;
	int height;
	int width;
	
	public Segment(int x, int y, int width, int height) {
		this.x = x;
		this.y = y;
		this.width = width;
		this.height = height;
	}
	
	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + x;
		result = prime * result + y;
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Segment other = (Segment) obj;
		if (x != other.x)
			return false;
		if (y != other.y)
			return false;
		return true;
	}
	
	public int getX() {
		return x;
	}
	public void setX(int x) {
		this.x = x;
	}
	public int getY() {
		return y;
	}
	public void setY(int y) {
		this.y = y;
	}
	public int getHeight() {
		return height;
	}
	public void setHeight(int height) {
		this.height = height;
	}
	public int getWidth() {
		return width;
	}
	public void setWidth(int width) {
		this.width = width;
	}

	@Override
	public String toString() {
		return "Segment [x=" + x + ", y=" + y + ", height=" + height + ", width=" + width + "]";
	}

}

class SegmentComparator implements Comparator<Segment> {

	@Override
	public int compare(Segment s1, Segment s2) {
		if (s1.x == s2.x)
			return 0;
		else if ((s1.x > s2.x))
			return 1;
		else
			return -1;
	}
}  
